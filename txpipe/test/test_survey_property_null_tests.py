import numpy as np
import healpy as hp
import healsparse as hsp

from ..survey_property_null_tests import (
    _in_footprint,
    _compute_bin_edges,
    _property_values_at,
    _marker_offset,
)

# A small footprint: a disc of radius 10 deg centred at (ra, dec) = (30, 10),
# NEST ordering to match the stage.
NSIDE_COV = 32
NSIDE = 256
CENTER = hp.ang2vec(30.0, 10.0, lonlat=True)


def _make_mask():
    """Boolean HealSparse footprint mask, as read_mask(returnbool=True) gives."""
    fp_pix = hp.query_disc(NSIDE, CENTER, np.radians(10.0), nest=True)
    mask = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.bool_, sentinel=False)
    mask[fp_pix] = True
    return mask, fp_pix


def _positions(pixels):
    """(ra, dec) at the centres of the given NEST pixels."""
    return hp.pix2ang(NSIDE, np.asarray(pixels), nest=True, lonlat=True)


# ---------------------------------------------------------------------------
# _in_footprint
# ---------------------------------------------------------------------------
def test_in_footprint_center_and_far():
    mask, _ = _make_mask()
    assert _in_footprint(np.array([30.0]), np.array([10.0]), mask)[0]
    assert not _in_footprint(np.array([200.0]), np.array([-40.0]), mask)[0]


def test_in_footprint_matches_direct_membership():
    mask, fp_pix = _make_mask()
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 60, size=5000)
    dec = rng.uniform(-10, 30, size=5000)
    flag = _in_footprint(ra, dec, mask)
    ref_pix = hp.ang2pix(NSIDE, ra, dec, lonlat=True, nest=True)
    assert np.array_equal(flag, np.isin(ref_pix, fp_pix))
    # Every flagged position must sit on a True mask pixel.
    assert mask[ref_pix[flag]].all()


# ---------------------------------------------------------------------------
# _compute_bin_edges
# ---------------------------------------------------------------------------
def test_compute_bin_edges_restricts_to_footprint():
    mask, fp_pix = _make_mask()
    # Property map covers a larger disc; values inside the footprint are in
    # [0, 1), values outside (but still covered) are huge. Only the in-footprint
    # values must set the range.
    big_pix = hp.query_disc(NSIDE, CENTER, np.radians(25.0), nest=True)
    big_ra, big_dec = _positions(big_pix)
    in_fp = _in_footprint(big_ra, big_dec, mask)

    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.float64)
    rng = np.random.default_rng(1)
    vals = np.empty(big_pix.size)
    vals[in_fp] = rng.uniform(0.0, 1.0, size=in_fp.sum())
    vals[~in_fp] = rng.uniform(1000.0, 2000.0, size=(~in_fp).sum())
    prop[big_pix] = vals

    edges = _compute_bin_edges(prop, NSIDE, 20, 0.05, mask)
    assert edges is not None
    assert edges.size == 21
    assert np.all(np.diff(edges) > 0)
    # Out-of-footprint values (>=1000) must not stretch the range.
    assert edges[-1] < 2.0
    # 5%/95% trim of Uniform(0,1) lands near [0.05, 0.95].
    assert 0.0 < edges[0] < 0.15
    assert 0.85 < edges[-1] < 1.0


def _selected_by_strict_bins(vals, edges):
    """How many values MeanShearInBins.selector would put into some bin.

    The selector is strict on both sides, so a value landing exactly on an edge
    belongs to no bin at all.
    """
    return sum(
        ((vals > edges[i]) & (vals < edges[i + 1])).sum()
        for i in range(edges.size - 1)
    )


def test_compute_bin_edges_integer_map_selects_every_value():
    """Integer maps (bright_objects/count, flag maps) must not land on edges.

    With evenly spaced edges every integer falls exactly on one, the strict
    selector drops all of them, and the whole null test comes out empty.
    """
    mask, fp_pix = _make_mask()
    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.int32)
    # Counts 0..4, as a bright-object map masked to small values would give.
    counts = np.arange(fp_pix.size) % 5
    prop[fp_pix] = counts.astype(np.int32)

    edges = _compute_bin_edges(prop, NSIDE, 20, 0.05, mask)
    assert edges is not None
    assert np.all(np.diff(edges) > 0)

    vals = np.unique(counts).astype(float)
    # One bin per distinct value, each value strictly inside its own bin.
    assert edges.size == vals.size + 1
    assert _selected_by_strict_bins(vals, edges) == vals.size
    for v in vals:
        assert ((v > edges[:-1]) & (v < edges[1:])).sum() == 1


def test_compute_bin_edges_none_when_constant():
    """A map that is constant inside the footprint carries no trend to test."""
    mask, fp_pix = _make_mask()
    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.float64)
    prop[fp_pix] = 3.0
    assert _compute_bin_edges(prop, NSIDE, 20, 0.05, mask) is None


def test_compute_bin_edges_continuous_map_stays_evenly_spaced():
    """Continuous maps keep the original evenly spaced binning."""
    mask, fp_pix = _make_mask()
    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.float64)
    prop[fp_pix] = np.random.default_rng(3).uniform(0.0, 1.0, size=fp_pix.size)

    edges = _compute_bin_edges(prop, NSIDE, 20, 0.05, mask)
    assert edges.size == 21
    assert np.allclose(np.diff(edges), edges[1] - edges[0])


def test_compute_bin_edges_none_when_disjoint():
    mask, _ = _make_mask()
    far_pix = hp.query_disc(
        NSIDE, hp.ang2vec(200.0, -40.0, lonlat=True), np.radians(5.0), nest=True
    )
    far = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.float64)
    far[far_pix] = np.random.default_rng(2).uniform(0, 1, size=far_pix.size)
    assert _compute_bin_edges(far, NSIDE, 20, 0.05, mask) is None


# ---------------------------------------------------------------------------
# _property_values_at
#
# The masking must be correct for every map dtype: unobserved pixels are found
# via HealSparse's validity mask, not by comparing to hp.UNSEEN (whose sentinel
# differs for integer and float32 maps).
# ---------------------------------------------------------------------------
def _three_galaxy_positions(mask, fp_pix, map_pix):
    """Positions of three galaxies:
    A = in footprint, on an observed map pixel   (should keep its value)
    B = in footprint, on an UNobserved map pixel (should become NaN)
    C = outside the footprint                     (should become NaN)
    """
    pix_a = map_pix[0]
    pix_b = np.setdiff1d(fp_pix, map_pix)[0]
    far = hp.query_disc(
        NSIDE, hp.ang2vec(200.0, -40.0, lonlat=True), np.radians(5.0), nest=True
    )
    pix_c = far[0]
    ra, dec = _positions([pix_a, pix_b, pix_c])
    fp = _in_footprint(ra, dec, mask)
    assert list(fp) == [True, True, False]
    return ra, dec, fp


def _run_dtype(dtype, value):
    mask, fp_pix = _make_mask()
    # Observed pixels = a smaller disc, leaving footprint pixels the property
    # map does not observe (the B case).
    map_pix = hp.query_disc(NSIDE, CENTER, np.radians(6.0), nest=True)
    ra, dec, fp = _three_galaxy_positions(mask, fp_pix, map_pix)

    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, dtype)
    prop[map_pix] = value

    vals = _property_values_at(prop, NSIDE, ra, dec, fp)
    # A keeps the observed value; B and C are NaN regardless of dtype.
    assert vals[0] == float(value)
    assert np.isnan(vals[1])
    assert np.isnan(vals[2])


def test_property_values_at_float64():
    _run_dtype(np.float64, 5.0)


def test_property_values_at_int32():
    # Integer maps (e.g. depth_count, flags/flag_*): the old == hp.UNSEEN test
    # missed these because the int sentinel is INT_MIN, not hp.UNSEEN.
    _run_dtype(np.int32, 7)


def test_property_values_at_int64():
    _run_dtype(np.int64, 7)


def test_property_values_at_float32():
    # float32 maps (e.g. external DP1): float32(UNSEEN) upcast != float64 UNSEEN.
    _run_dtype(np.float32, 5.0)


def test_property_values_at_inrange_sentinel():
    """A map whose unobserved sentinel falls inside the value range: the old
    == hp.UNSEEN masking would leak the unobserved galaxy; the validity-mask
    approach must still NaN it."""
    mask, fp_pix = _make_mask()
    map_pix = hp.query_disc(NSIDE, CENTER, np.radians(6.0), nest=True)
    ra, dec, fp = _three_galaxy_positions(mask, fp_pix, map_pix)

    # sentinel 5.0 sits right among the plausible bin values; real data ~1.0.
    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.float64, sentinel=5.0)
    prop[map_pix] = 1.0

    vals = _property_values_at(prop, NSIDE, ra, dec, fp)
    assert vals[0] == 1.0
    assert np.isnan(vals[1])  # unobserved -> NaN even though sentinel is in range
    assert np.isnan(vals[2])


# ---------------------------------------------------------------------------
# _marker_offset
#
# mu holds the mean property value per bin, so the points are unevenly spaced.
# The offset must stay inside the narrowest gap, or the g1 and g2 series cross
# into neighbouring bins and are drawn out of order.
# ---------------------------------------------------------------------------
def _assert_no_crossing(mu):
    """g1 sits at mu - dx and g2 at mu + dx; neighbours must not interleave."""
    dx = _marker_offset(mu)
    mu = np.sort(mu)
    assert np.all(mu[:-1] + dx < mu[1:] - dx)


def test_marker_offset_fits_inside_narrowest_gap():
    # Two bins very close together, the rest far apart: an offset scaled from
    # the median gap would be many times too wide for the narrow one.
    mu = np.array([0.02, 0.05, 1.0, 2.0, 3.0, 5.0])
    gaps = np.diff(np.sort(mu))
    dx = _marker_offset(mu)
    assert 2 * dx < gaps.min()
    assert 2 * dx < 0.1 * np.median(gaps)  # strictly tighter than the old rule
    _assert_no_crossing(mu)


def test_marker_offset_evenly_spaced():
    mu = np.linspace(0.0, 1.0, 21)
    assert np.isclose(_marker_offset(mu), 0.1 * 0.05)
    _assert_no_crossing(mu)


def test_marker_offset_unsorted_input():
    """The offset must not depend on the order the bins arrive in."""
    mu = np.array([3.0, 0.02, 5.0, 0.05, 2.0, 1.0])
    assert np.isclose(_marker_offset(mu), _marker_offset(np.sort(mu)))


def test_marker_offset_degenerate_cases():
    # Nothing to space out: no offset, and no crash.
    assert _marker_offset(np.array([])) == 0.0
    assert _marker_offset(np.array([1.0])) == 0.0
    assert _marker_offset(np.array([2.0, 2.0, 2.0])) == 0.0


def test_marker_offset_integer_valued_bins():
    """Discrete maps give near-integer bin means; spacing must still work."""
    mu = np.array([0.0, 1.0, 2.0])
    assert np.isclose(_marker_offset(mu), 0.1)
    _assert_no_crossing(mu)
