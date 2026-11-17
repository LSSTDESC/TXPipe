import numpy as np
import healpy as hp
import healsparse as hsp

from ..survey_property_null_tests import (
    _bin_edges_from_values,
    _counts_in_bins,
    _property_values_at,
    _assign_property_column,
    _marker_offset,
    _min_count_cut_report,
    _bad_bin_report,
    _null_test_statistics,
    _band_of,
    _shorten_map_name,
    _hs_map_names,
    _keep_psf_map,
)
from ..calibration.calibration_utils.names import META_VARIANTS

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


def test_bin_edges_integer_counts_lose_no_galaxies():
    """Integer maps with more values than bins: no value may sit on an edge."""
    rng = np.random.default_rng(3)
    for mean in (40, 400):
        vals = rng.poisson(mean, 100_000).astype(float)
        edges = _bin_edges_from_values(vals, 20, 0.05)
        assert not np.isin(vals, edges).any()
        lo = np.percentile(vals, 5)
        hi = np.percentile(vals, 95)
        inside = vals[(vals >= lo) & (vals <= hi)]
        # Every value between the 5th and 95th percentiles is in exactly one bin.
        assert _counts_in_bins(inside, edges).sum() == inside.size
        assert np.all(np.diff(edges) > 0)


def test_bin_edges_pixelized_float_map_loses_no_galaxies():
    """Galaxies share their pixel's value, so a float map can hit an edge too."""
    rng = np.random.default_rng(5)
    pixel_values = rng.normal(2000.0, 50.0, 3000)
    vals = pixel_values[rng.integers(0, pixel_values.size, 100_000)]
    edges = _bin_edges_from_values(vals, 20, 0.05)
    assert not np.isin(vals, edges).any()
    inside = vals[(vals >= np.percentile(vals, 5)) & (vals <= np.percentile(vals, 95))]
    assert _counts_in_bins(inside, edges).sum() == inside.size


def test_bin_edges_float_map_unchanged_by_snapping():
    vals = np.random.default_rng(4).normal(2000.0, 50.0, 50_000)
    edges = _bin_edges_from_values(vals, 20, 0.05)
    expected = np.linspace(np.percentile(vals, 5), np.percentile(vals, 95), 21)
    assert np.array_equal(edges, expected)


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


def test_property_values_at_empty_input():
    """An empty chunk gives an empty float array for every map dtype.

    combined_iterators(longest=True) keeps yielding chunks after the shorter
    input runs out, so a chunk can have no rows. HealSparse returns an empty
    array of the map's own dtype for no pixels, ignoring valid_mask, which
    made ``~valid`` fail ("ufunc 'invert' not supported") on the first DP2 run.
    """
    empty = np.empty(0, dtype=float)
    for dtype, value in [(np.float64, 5.0), (np.float32, 5.0), (np.int32, 7), (np.int64, 7)]:
        prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, dtype)
        prop[_patch_pixels()] = value

        vals = _property_values_at(prop, NSIDE, empty, empty)
        assert vals.shape == (0,)
        assert vals.dtype == np.float64
        # The bin-edge pass filters the values like this; it must work on empty too.
        assert vals[np.isfinite(vals)].shape == (0,)


# ---------------------------------------------------------------------------
# _assign_property_column
#
# Regression for the metadetect broadcast bug: each shear variant is a separate
# detection catalogue with its own length, and MeanShearInBins runs the bin
# selector on every variant. A single unsheared-length survey_prop column cannot
# be broadcast against the per-variant bin_1p etc. columns, which raised
# "operands could not be broadcast together". _assign_property_column must write
# one property column per variant, each matching that variant's length.
# ---------------------------------------------------------------------------
def _observed_map(value=1.0):
    prop = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE, np.float64)
    prop[_patch_pixels()] = value
    return prop


def test_assign_property_column_metadetect_per_variant_lengths():
    prop = _observed_map(1.0)
    patch_pix = _patch_pixels()
    rng = np.random.default_rng(0)

    # Give every variant a DIFFERENT length, like the real (48297 vs 48673) case.
    data = {}
    lengths = {}
    for i, v in enumerate(META_VARIANTS):
        n = 100 + 13 * i  # ns=100, 1p=113, 1m=126, ... all distinct
        ra, dec = _positions(rng.choice(patch_pix, size=n))
        data[f"{v}/ra"], data[f"{v}/dec"] = ra, dec
        lengths[v] = n

    _assign_property_column(data, "metadetect", prop, NSIDE, "ns/ra", "ns/dec")

    # One property column per variant, each at that variant's own length.
    for v in META_VARIANTS:
        assert data[f"{v}/survey_prop"].shape == (lengths[v],)

    # The plain column MeanShearInBins.add_data indexes mirrors the unsheared one.
    assert data["survey_prop"].shape == (lengths["ns"],)
    assert np.array_equal(
        data["survey_prop"], data[f"{META_VARIANTS[0]}/survey_prop"], equal_nan=True
    )

    # The selection that used to crash now broadcasts for every variant:
    # (survey_prop in range) & (per-variant bin != -1).
    for v in META_VARIANTS:
        bin_col = np.zeros(lengths[v], dtype=int)
        w = (data[f"{v}/survey_prop"] > 0.0) & (bin_col != -1)
        assert w.shape == (lengths[v],)


def test_assign_property_column_scalar_metadetect_has_three_variants():
    prop = _observed_map(1.0)
    patch_pix = _patch_pixels()
    rng = np.random.default_rng(1)
    variants = ["ns", "1p", "1m"]
    data = {}
    for i, v in enumerate(variants):
        ra, dec = _positions(rng.choice(patch_pix, size=50 + 7 * i))
        data[f"{v}/ra"], data[f"{v}/dec"] = ra, dec

    _assign_property_column(data, "scalar_metadetect", prop, NSIDE, "ns/ra", "ns/dec")

    for i, v in enumerate(variants):
        assert data[f"{v}/survey_prop"].shape == (50 + 7 * i,)
    assert "2p/survey_prop" not in data
    assert data["survey_prop"].shape == (50,)


def test_assign_property_column_single_for_non_metadetect():
    """metacal/lensfit/hsc share one length across variants: one column only."""
    prop = _observed_map(2.5)
    ra, dec = _positions(_patch_pixels()[:200])
    data = {"ra": ra, "dec": dec}

    _assign_property_column(data, "metacal", prop, NSIDE, "ra", "dec")

    assert data["survey_prop"].shape == (200,)
    assert [k for k in data if "survey_prop" in k] == ["survey_prop"]


def test_assign_property_column_metadetect_empty_variants():
    """Late chunks from combined_iterators(longest=True): the variants have
    different lengths, so the shorter ones run out first and arrive empty while
    a longer one still has rows. Every variant must still get a property column
    of its own length, including zero."""
    prop = _observed_map(1.0)
    patch_pix = _patch_pixels()
    rng = np.random.default_rng(2)

    # ns and most variants exhausted; only 1p still has rows.
    lengths = {v: 0 for v in META_VARIANTS}
    lengths["1p"] = 5
    data = {}
    for v in META_VARIANTS:
        ra, dec = _positions(rng.choice(patch_pix, size=lengths[v]))
        data[f"{v}/ra"], data[f"{v}/dec"] = ra, dec
        data[f"{v}/bin"] = np.zeros(lengths[v], dtype=int)

    _assign_property_column(data, "metadetect", prop, NSIDE, "ns/ra", "ns/dec")

    for v in META_VARIANTS:
        assert data[f"{v}/survey_prop"].shape == (lengths[v],)
        assert data[f"{v}/survey_prop"].dtype == np.float64
    assert data["survey_prop"].shape == (0,)
    assert np.all(data["1p/survey_prop"] == 1.0)

    # The bin-edge pass's selection on the unsheared variant, on an empty chunk.
    in_source_bin = data["ns/bin"] != -1
    vals = data["ns/survey_prop"]
    assert vals[in_source_bin & np.isfinite(vals)].shape == (0,)


def test_assign_property_column_empty_chunk_non_metadetect():
    """An empty chunk for a single-position catalog type (metacal etc.)."""
    prop = _observed_map(2.5)
    empty = np.empty(0, dtype=float)
    data = {"ra": empty, "dec": empty}

    _assign_property_column(data, "metacal", prop, NSIDE, "ra", "dec")

    assert data["survey_prop"].shape == (0,)
    assert data["survey_prop"].dtype == np.float64


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


# Map naming
#
# DP1 ingestion writes deepCoadd_<property>_consolidated_map_<reduction>_<band>.hs
# for every property in all six bands; these names become plot titles and HDF5
# group names, so the stage trims them down.


def test_shorten_drops_the_constant_dp1_parts():
    assert (
        _shorten_map_name("deepCoadd_psf_size_consolidated_map_weighted_mean_i")
        == "psf_size_i"
    )
    assert (
        _shorten_map_name("deepCoadd_sky_background_consolidated_map_weighted_mean_r")
        == "sky_background_r"
    )


def test_shorten_keeps_the_discriminating_reduction():
    """The three epoch maps differ only in their reduction, so it must survive."""
    names = {
        _shorten_map_name(f"deepCoadd_epoch_consolidated_map_{r}_i")
        for r in ("min", "max", "mean")
    }
    assert names == {"epoch_min_i", "epoch_max_i", "epoch_mean_i"}
    assert (
        _shorten_map_name("deepCoadd_exposure_time_consolidated_map_sum_i")
        == "exposure_time_sum_i"
    )


def test_shorten_leaves_non_dp1_names_alone():
    """SUPREME maps come in through external_maps_dir and keep their names."""
    for name in ("supreme_dc2_dr6d_v2_airmass_i", "depth", "psf_size"):
        assert _shorten_map_name(name) == name


def test_hs_map_names_falls_back_to_stems_on_collision():
    """A shortened name that clashes would otherwise drop a map silently."""
    paths = [
        "/d/deepCoadd_x_consolidated_map_weighted_mean_i.hs",  # -> x_i
        "/d/x_i.hs",  # already x_i
    ]
    names = _hs_map_names(paths)
    assert set(names.values()) == {
        "deepCoadd_x_consolidated_map_weighted_mean_i",
        "x_i",
    }


def test_hs_map_names_unique_across_the_dp1_set():
    dp1 = [
        "deepCoadd_exposure_time_consolidated_map_sum",
        "deepCoadd_epoch_consolidated_map_min",
        "deepCoadd_epoch_consolidated_map_max",
        "deepCoadd_epoch_consolidated_map_mean",
        "deepCoadd_psf_size_consolidated_map_weighted_mean",
        "deepCoadd_psf_e1_consolidated_map_weighted_mean",
        "deepCoadd_psf_e2_consolidated_map_weighted_mean",
        "deepCoadd_psf_maglim_consolidated_map_weighted_mean",
        "deepCoadd_sky_background_consolidated_map_weighted_mean",
        "deepCoadd_sky_noise_consolidated_map_weighted_mean",
        "deepCoadd_dcr_dra_consolidated_map_weighted_mean",
        "deepCoadd_dcr_ddec_consolidated_map_weighted_mean",
        "deepCoadd_dcr_e1_consolidated_map_weighted_mean",
        "deepCoadd_dcr_e2_consolidated_map_weighted_mean",
    ]
    paths = [f"/d/{m}_{b}.hs" for m in dp1 for b in "ugrizy"]
    names = _hs_map_names(paths)
    # No collisions, so nothing fell back to a full stem.
    assert len(set(names.values())) == len(paths)
    assert names[f"/d/{dp1[4]}_i.hs"] == "psf_size_i"


def test_band_of():
    for b in "ugrizy":
        assert _band_of(f"deepCoadd_psf_size_consolidated_map_weighted_mean_{b}") == b
    # Not a band: no suffix, a multi-letter tail, or a letter outside ugrizy.
    assert _band_of("depth") is None
    assert _band_of("deepCoadd_epoch_consolidated_map_mean") is None
    assert _band_of("some_map_x") is None


def test_band_filter_keeps_maps_with_no_band_suffix():
    """The bands cut must not silently discard non-DP1 maps."""
    bands = ["i"]
    stems = ["psf_size_i", "psf_size_r", "supreme_airmass"]
    kept = [s for s in stems if _band_of(s) in (None, *bands)]
    assert kept == ["psf_size_i", "supreme_airmass"]


# ---------------------------------------------------------------------------
# _min_count_cut_report
# ---------------------------------------------------------------------------
def test_min_count_cut_report_none_when_nothing_cut():
    edges = np.linspace(0.0, 1.0, 5)
    assert _min_count_cut_report("depth", edges, np.array([50, 80, 90, 60]), 50) is None


def test_min_count_cut_report_lists_each_cut_bin():
    """Bins below min_count, including empty ones, are each listed with their
    range and count; a bin at exactly min_count is kept, matching the >= cut in
    _null_test_statistics."""
    edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    counts = np.array([12, 50, 300, 0])
    msg = _min_count_cut_report("sky_background_i", edges, counts, 50)
    assert "'sky_background_i': 2 of 4 bins" in msg
    assert "bin 0 (0 to 0.25): 12 galaxies" in msg
    assert "bin 3 (0.75 to 1): 0 galaxies" in msg
    assert "bin 1 " not in msg and "bin 2 " not in msg


# ---------------------------------------------------------------------------
# _bad_bin_report
# ---------------------------------------------------------------------------
def _good_bins(n=4):
    edges = np.linspace(0.0, 1.0, n + 1)
    mu = 0.5 * (edges[1:] + edges[:-1])
    g1 = np.full(n, 1e-3)
    g2 = np.full(n, -1e-3)
    s1 = np.full(n, 1e-2)
    s2 = np.full(n, 1e-2)
    counts = np.full(n, 1000)
    return edges, mu, g1, g2, s1, s2, counts


def test_bad_bin_report_none_for_healthy_bins():
    assert _bad_bin_report("depth", *_good_bins(), 50) is None


def test_bad_bin_report_flags_populated_nan_and_zero_error_bins():
    """A well-populated bin with a NaN shear (e.g. one NaN in the catalog) or a
    zero error is reported; healthy bins are not."""
    edges, mu, g1, g2, s1, s2, counts = _good_bins()
    g1[1] = np.nan
    s2[3] = 0.0
    msg = _bad_bin_report("psf_size_i", edges, mu, g1, g2, s1, s2, counts, 50)
    assert "WARNING - 'psf_size_i': 2 bin(s)" in msg
    assert "bin 1 (0.25 to 0.5): 1000 galaxies, g1=nan" in msg
    assert "bin 3 (0.75 to 1): 1000 galaxies" in msg and "sigma_g2=0" in msg
    assert "bin 0 " not in msg and "bin 2 " not in msg


def test_bad_bin_report_ignores_small_bins():
    """Empty or tiny bins are NaN for expected reasons; the count cut already
    handles them, so they are not reported here."""
    edges, mu, g1, g2, s1, s2, counts = _good_bins()
    counts[0], counts[3] = 0, 3
    for arr in (mu, g1, g2, s1, s2):
        arr[0] = np.nan
    s1[3] = 0.0
    assert _bad_bin_report("depth", edges, mu, g1, g2, s1, s2, counts, 50) is None


# ---------------------------------------------------------------------------
# _null_test_statistics: chi2 against zero and against a constant
# ---------------------------------------------------------------------------
def _stats_inputs(n=20, sigma=1e-3):
    mu = np.linspace(0.0, 1.0, n)
    s = np.full(n, sigma)
    counts = np.full(n, 1000)
    return mu, s, counts


def test_chi2_const_removes_a_constant_offset():
    """A pure offset (same shear in every bin) fails the test against zero but
    gives chi2_const = 0, with the offset recovered as const_g1/const_g2."""
    mu, s, counts = _stats_inputs()
    g1 = np.full(mu.size, 3e-3)
    g2 = np.full(mu.size, -2e-3)
    _, st = _null_test_statistics(mu, g1, g2, s, s, counts, 50)

    # Against zero: 20 bins x ((3)^2 + (2)^2) = 260.
    assert np.isclose(st["chi2_null"], 20 * (3.0**2 + 2.0**2))
    assert st["n_dof_null"] == 40
    assert np.isclose(st["chi2_const"], 0.0, atol=1e-9)
    assert st["n_dof_const"] == 38
    assert np.isclose(st["chi2_const_pvalue"], 1.0)
    assert np.isclose(st["const_g1"], 3e-3)
    assert np.isclose(st["const_g2"], -2e-3)
    assert np.isclose(st["const_g1_err"], 1e-3 / np.sqrt(20))


def test_chi2_const_matches_direct_sum_and_skips_unused_bins():
    """chi2_const equals the weighted residual sum about the weighted mean,
    over the used bins only (a low-count bin with a wild value is ignored)."""
    rng = np.random.default_rng(4)
    mu, _, counts = _stats_inputs()
    s1 = rng.uniform(0.5e-3, 2e-3, mu.size)
    s2 = rng.uniform(0.5e-3, 2e-3, mu.size)
    g1 = rng.normal(1e-3, s1)
    g2 = rng.normal(0.0, s2)
    counts[5] = 10          # cut by min_count
    g1[5] = 1.0             # would dominate if it were used

    usable, st = _null_test_statistics(mu, g1, g2, s1, s2, counts, 50)
    expected = 0.0
    for g, s in ((g1, s1), (g2, s2)):
        w = 1 / s[usable] ** 2
        m = np.sum(w * g[usable]) / np.sum(w)
        expected += np.sum(w * (g[usable] - m) ** 2)
    assert not usable[5]
    assert np.isclose(st["chi2_const"], expected)
    assert st["n_dof_const"] == 2 * 19 - 2


def test_chi2_const_dof_match_its_mean_under_the_null():
    """With Gaussian noise about an offset, the mean chi2_const over many
    realisations equals its dof (2n - 2), confirming the dof count."""
    rng = np.random.default_rng(5)
    mu, s, counts = _stats_inputs(n=10)
    vals = []
    for _ in range(4000):
        g1 = 2e-3 + rng.normal(0.0, s)
        g2 = -1e-3 + rng.normal(0.0, s)
        vals.append(_null_test_statistics(mu, g1, g2, s, s, counts, 50)[1]["chi2_const"])
    # dof = 18; the mean of 4000 draws has a standard error of sqrt(36/4000) ~ 0.1.
    assert abs(np.mean(vals) - 18) < 0.5


def test_chi2_const_needs_two_bins():
    """One used bin leaves no freedom after fitting the constant: NaN, dof 0.
    No used bins: everything NaN, nothing raised."""
    mu, s, counts = _stats_inputs(n=3)
    g = np.full(3, 1e-3)
    counts[1:] = 0
    _, st = _null_test_statistics(mu, g, g, s, s, counts, 50)
    assert st["n_used_bins"] == 1
    assert st["n_dof_const"] == 0 and np.isnan(st["chi2_const"])
    assert np.isnan(st["chi2_const_pvalue"])
    assert np.isfinite(st["const_g1"])

    counts[:] = 0
    with np.errstate(all="raise"):
        _, st = _null_test_statistics(mu, g, g, s, s, counts, 50)
    assert st["n_used_bins"] == 0
    assert np.isnan(st["chi2_const"]) and np.isnan(st["const_g1"])


def test_keep_psf_map_default_keeps_only_2d():
    names = ["psf/g1_0", "psf/g1_3", "psf/g1_2D", "psf/count_2D", "psf/var_e_1"]
    kept = [n for n in names if _keep_psf_map(n, ["2D"])]
    assert kept == ["psf/g1_2D", "psf/count_2D"]


def test_keep_psf_map_empty_list_keeps_all_and_ints_match():
    assert _keep_psf_map("psf/g1_0", [])
    assert _keep_psf_map("psf/g1_0", [0])
    assert not _keep_psf_map("psf/g1_10", [0])

