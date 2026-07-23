import numpy as np
import healpy as hp
import healsparse as hsp

from ..survey_property_null_tests import (
    _bin_edges_from_values,
    _property_values_at,
    _marker_offset,
)

# A small patch: a disc of radius 10 deg centred at (ra, dec) = (30, 10),
# NEST ordering to match the stage.
NSIDE_COV = 32
NSIDE = 256
CENTER = hp.ang2vec(30.0, 10.0, lonlat=True)


def _patch_pixels():
    """NEST pixels of the disc used as the observed region in the tests."""
    return hp.query_disc(NSIDE, CENTER, np.radians(10.0), nest=True)


def _positions(pixels):
    """(ra, dec) at the centres of the given NEST pixels."""
    return hp.pix2ang(NSIDE, np.asarray(pixels), nest=True, lonlat=True)


# ---------------------------------------------------------------------------
# _bin_edges_from_values
#
# The edges are now chosen from the property values sampled at the galaxies that
# enter the measurement (source-bin galaxies on observed pixels), which the run
# method assembles. These tests drive the pure edge logic on such value arrays;
# the observed-pixel restriction that produces them is exercised through
# _property_values_at.
# ---------------------------------------------------------------------------
def _selected_by_strict_bins(vals, edges):
    """How many values MeanShearInBins.selector would put into some bin.

    The selector is strict on both sides, so a value landing exactly on an edge
    belongs to no bin at all.
    """
    return sum(
        ((vals > edges[i]) & (vals < edges[i + 1])).sum()
        for i in range(edges.size - 1)
    )


def test_bin_edges_continuous_values_evenly_spaced():
    """Continuous values give evenly spaced edges spanning the trimmed range."""
    vals = np.random.default_rng(3).uniform(0.0, 1.0, size=20000)
    edges = _bin_edges_from_values(vals, 20, 0.05)
    assert edges.size == 21
    assert np.allclose(np.diff(edges), edges[1] - edges[0])
    # 5%/95% trim of Uniform(0,1) lands near [0.05, 0.95].
    assert 0.0 < edges[0] < 0.15
    assert 0.85 < edges[-1] < 1.0


def test_bin_edges_span_only_the_sampled_values():
    """The edges come purely from the values passed in: values a caller has
    already excluded (e.g. galaxies off the footprint or on unobserved pixels)
    cannot stretch the range, because they never reach this function."""
    rng = np.random.default_rng(1)
    # The sample the measurement actually bins: all in [0, 1).
    vals = rng.uniform(0.0, 1.0, size=20000)
    edges = _bin_edges_from_values(vals, 20, 0.05)
    assert edges is not None
    assert edges.size == 21
    assert np.all(np.diff(edges) > 0)
    assert edges[-1] < 2.0


def test_bin_edges_integer_values_select_every_value():
    """Integer-valued samples (bright_objects/count, flag maps) must not land on
    edges. With evenly spaced edges every integer falls exactly on one, the
    strict selector drops all of them, and the whole null test comes out empty.
    """
    counts = (np.arange(50000) % 5).astype(np.int32)
    edges = _bin_edges_from_values(counts.astype(float), 20, 0.05)
    assert edges is not None
    assert np.all(np.diff(edges) > 0)

    vals = np.unique(counts).astype(float)
    # One bin per distinct value, each value strictly inside its own bin.
    assert edges.size == vals.size + 1
    assert _selected_by_strict_bins(vals, edges) == vals.size
    for v in vals:
        assert ((v > edges[:-1]) & (v < edges[1:])).sum() == 1


def test_bin_edges_none_when_constant():
    """A property that is constant across the sample carries no trend to test."""
    assert _bin_edges_from_values(np.full(1000, 3.0), 20, 0.05) is None


def test_bin_edges_none_when_empty():
    """No galaxy sampled the map: nothing to bin, so the map is skipped."""
    assert _bin_edges_from_values(np.empty(0), 20, 0.05) is None


# ---------------------------------------------------------------------------
# _property_values_at
#
# The masking must be correct for every map dtype: unobserved pixels are found
# via HealSparse's validity mask, not by comparing to hp.UNSEEN (whose sentinel
# differs for integer and float32 maps).
# ---------------------------------------------------------------------------
def _two_galaxy_positions(patch_pix, map_pix):
    """Positions of two galaxies:
    A = on an observed map pixel   (should keep its value)
    B = on an UNobserved map pixel (should become NaN)

    A galaxy well outside the map's observed area lands on an unobserved pixel
    too, so it is covered by the same B case.
    """
    pix_a = map_pix[0]
    pix_b = np.setdiff1d(patch_pix, map_pix)[0]
    ra, dec = _positions([pix_a, pix_b])
    return ra, dec


def _run_dtype(dtype, value):
    patch_pix = _patch_pixels()
    # Observed pixels = a smaller disc, leaving patch pixels the property map
    # does not observe (the B case).
    map_pix = hp.query_disc(NSIDE, CENTER, np.radians(6.0), nest=True)
    ra, dec = _two_galaxy_positions(patch_pix, map_pix)

    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, dtype)
    prop[map_pix] = value

    vals = _property_values_at(prop, NSIDE, ra, dec)
    # A keeps the observed value; B is NaN regardless of dtype.
    assert vals[0] == float(value)
    assert np.isnan(vals[1])


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
    patch_pix = _patch_pixels()
    map_pix = hp.query_disc(NSIDE, CENTER, np.radians(6.0), nest=True)
    ra, dec = _two_galaxy_positions(patch_pix, map_pix)

    # sentinel 5.0 sits right among the plausible bin values; real data ~1.0.
    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.float64, sentinel=5.0)
    prop[map_pix] = 1.0

    vals = _property_values_at(prop, NSIDE, ra, dec)
    assert vals[0] == 1.0
    assert np.isnan(vals[1])  # unobserved -> NaN even though sentinel is in range


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
