import collections
import glob
import os
import pathlib
import numpy as np
from ceci.config import StageParameter

from .base_stage import PipelineStage
from .data_types import (
    ShearCatalog,
    TomographyCatalog,
    MapsFile,
    FileCollection,
    HDFFile,
    class_for_shear_catalog_type,
)
from .calibration.calibration_utils import MeanShearInBins
from .utils import read_shear_catalog_type

BANDS = "ugrizy"

# Fewest usable bins to attempt the straight line fit on.
_MIN_FIT_BINS = 3


def _hs_files(root):
    """
    List the .hs healsparse files under a survey property map root.

    ``root`` may be a directory, e.g. what the survey_property_maps FileCollection
    from ingestion stage, or a filename prefix ``root*.hs``.
    """
    if not root:
        return []
    pattern = f"{root.rstrip('/')}/*.hs" if os.path.isdir(root) else f"{root}*.hs"
    return sorted(glob.glob(pattern))


def _band_of(stem):
    """
    The band a survey property map ends in, or None if no band
    suffix. DP1 writes one file per band, named e.g. *_weighted_mean_i.
    """
    band = stem.rpartition("_")[2]
    return band if band in BANDS and len(band) == 1 else None


def _shorten_map_name(stem):
    """
    Trim a survey property filename down for plotting.

    E.g. replaces ``deepCoadd_<property>_consolidated_map_<reduction>_<band>``
    with <property>_<reduction>_<band>, such as epoch_max_i.

    Names without _consolidated_map in them are left alone, so SUPREME/DC2 maps
    keep the names they have always had.
    """
    if "_consolidated_map" not in stem:
        return stem
    name = stem.replace("_consolidated_map", "", 1)
    if name.startswith("deepCoadd_"):
        name = name[len("deepCoadd_"):]
    return name.replace("_weighted_mean", "", 1)


def _hs_map_names(paths):
    """
    Decide a shortened name for each .hs file and avoid clashes by keeping
    the full name.
    """
    stems = {path: pathlib.Path(path).stem for path in paths}
    short = {path: _shorten_map_name(stem) for path, stem in stems.items()}
    counts = collections.Counter(short.values())
    return {
        path: (stems[path] if counts[name] > 1 else name)
        for path, name in short.items()
    }


def _keep_psf_map(name, psf_map_bins):
    """
    Whether to test a psf_maps map, from its bin suffix (psf/g1_2D -> 2D).
    An empty psf_map_bins keeps every map.
    """
    return not psf_map_bins or name.rsplit("_", 1)[-1] in [str(b) for b in psf_map_bins]


def _bin_edges_from_values(vals, nbins, outlier_fraction):
    """
    On a first pass, compute reasonable bin edges for a survey property according
    to its footprint.

    ``vals`` is the survey property sampled within the mean shear measurement region,
    i.e. ``{galaxies with defined shear in a source bin} ∩ {pixels where that map is
    observed}``. Non-finite entries must already be removed by the caller.

    Edge case: Maps taking only a few distinct integer values (e.g. flags, if supplied)
    may get dropped due to strict inequalities with edges in MeanShearInBins.selector.
    We put bin edges between these integer values to keep them. With more values
    than bins, any edge that lands on a value is moved halfway between neighbouring
    values, for the same reason.

    Returns ``None`` if the sample has no usable range -- because no galaxy
    sampled the map, or because the property is constant across the sample and
    so carries no trend to test. That map is then skipped.
    """
    if vals.size == 0:
        return None

    lo = np.percentile(vals, 100 * outlier_fraction)
    hi = np.percentile(vals, 100 * (1 - outlier_fraction))

    # Discrete-valued map: put the edges between the values so each value sits
    # at a bin centre.
    uniq = np.unique(vals[(vals >= lo) & (vals <= hi)])
    if uniq.size < 2:
        # i.e. survey property is a single value across the sample.
        return None
    if uniq.size <= nbins:
        mids = 0.5 * (uniq[1:] + uniq[:-1])
        return np.concatenate(
            (
                [uniq[0] - 0.5 * (uniq[1] - uniq[0])],
                mids,
                [uniq[-1] + 0.5 * (uniq[-1] - uniq[-2])],
            )
        )

    edges = np.linspace(lo, hi, nbins + 1)
    # If an edge lands on a value (integer maps, or a pixel's value in a float
    # map), those galaxies would be in no bin, so we move each edge halfway
    # between two values; the outer edges move outwards to keep lo and hi.
    if np.isin(edges, uniq).any():
        halfway = np.concatenate((
            [uniq[0] - 0.5 * (uniq[1] - uniq[0])],
            0.5 * (uniq[1:] + uniq[:-1]),
            [uniq[-1] + 0.5 * (uniq[-1] - uniq[-2])],
        ))
        j = np.clip(np.searchsorted(halfway, edges), 1, halfway.size - 1)
        nearer_below = edges - halfway[j - 1] <= halfway[j] - edges
        snapped = np.where(nearer_below, halfway[j - 1], halfway[j])
        snapped[0], snapped[-1] = halfway[j[0] - 1], halfway[j[-1]]
        edges = np.unique(snapped)
    return edges


def _counts_in_bins(vals, edges):
    """
    Number of values in each bin, selected the way ``MeanShearInBins.selector``
    selects them: strictly inside both edges. ``np.histogram`` is not used, since it
    includes the lower edge.
    """
    return np.array(
        [np.count_nonzero((vals > lo) & (vals < hi)) for lo, hi in zip(edges[:-1], edges[1:])],
        dtype=np.int64,
    )


def _property_values_at(hsp_map, nside, ra, dec):
    """
    Look up a survey property map at (ra, dec) positions. Returns ``NaN`` for
    galaxies on pixels the map does not observe (these are not binned).

    Unobserved pixels are detected with HealSparse's valid_mask=True so that it
    can apply to all map types (e.g. int32, float64) and sentinels (e.g. hp.UNSEEN).
    """
    import healpy as hp

    # Chunks can be empty (combined_iterators with longest=True), and HealSparse
    # then ignores valid_mask, so we return an empty array directly.
    if len(ra) == 0:
        return np.empty(0, dtype=float)

    pixels = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    # valid_mask=True flags observed pixels for any map dtype.
    valid = hsp_map.get_values_pix(pixels, valid_mask=True)
    prop_vals = hsp_map[pixels].astype(float)
    prop_vals[~valid] = np.nan
    return prop_vals


def _detection_variants(cat_type):
    """
    Return the shear-variant group names (ns, 1p, ...) for metadetect-style
    catalogs (metadetect, scalar_metadetect) with distinct RA/Dec or return None
    (metacal, anacal, etc.). Names taken from .data_types.
    """
    # class: metadetect style (separate ra, dec) or metacal style?
    cat_class = class_for_shear_catalog_type(cat_type)
    if not cat_class.get_primary_catalog_prefix():
        return None
    return [p.rstrip("/") for p in cat_class.get_classifier_prefixes()]


def _assign_property_column(data, cat_type, hsp_map, nside, ra_key, dec_key):
    """
    Set the survey_prop column(s) that MeanShearInBins bins on. Metadetect
    variants have their own positions, so each gets its own column, plus a
    plain survey_prop copied from the unsheared one; other types need one column.
    """
    variants = _detection_variants(cat_type)
    if variants is not None:
        for v in variants:
            data[f"{v}/survey_prop"] = _property_values_at(
                hsp_map, nside, data[f"{v}/ra"], data[f"{v}/dec"]
            )
        data["survey_prop"] = data[f"{variants[0]}/survey_prop"]
    else:
        data["survey_prop"] = _property_values_at(
            hsp_map, nside, data[ra_key], data[dec_key]
        )


def _weighted_line_fit(x, y, sigma):
    """
    Weighted least-squares linear fit with intercept.

    Returns ``(slope, slope_err, intercept, intercept_err)``. All four are NaN
    when there are fewer than ``_MIN_FIT_BINS`` points or all ``x`` are equal,
    so a fit that could not be made is never written out as if it were real.
    """
    nan = (np.nan, np.nan, np.nan, np.nan)
    if x.size < _MIN_FIT_BINS:
        return nan
    w = 1.0 / sigma**2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    det = S * Sxx - Sx**2
    if not det > 0:
        return nan
    slope = (S * Sxy - Sx * Sy) / det
    intercept = (Sxx * Sy - Sx * Sxy) / det
    return slope, np.sqrt(S / det), intercept, np.sqrt(Sxx / det)


def _min_count_cut_report(name, edges, counts, min_count):
    """
    List the bins cut for having fewer than ``min_count`` galaxies, with their
    range and count, or None. These are usually the end bins, where a
    systematic is most likely, so the cut is reported in the log.
    """
    cut = np.flatnonzero(np.asarray(counts) < min_count)
    if cut.size == 0:
        return None
    parts = [
        f"bin {i} ({edges[i]:.4g} to {edges[i + 1]:.4g}): {int(counts[i])} galaxies"
        for i in cut
    ]
    return (
        f"TXMeanShearSurveyProperties: '{name}': {cut.size} of {len(counts)} bins "
        f"have fewer than {min_count} galaxies and are left out of the chi-squared "
        "and the fits: " + "; ".join(parts)
    )


def _bad_bin_report(name, edges, mu, g1, g2, sigma1, sigma2, counts, min_count):
    """
    Warn about bins with at least ``min_count`` galaxies that still cannot be
    used (non-finite mean or error, or zero error), or None. This points to a
    problem in the inputs, e.g. a NaN shear or weight in the catalog.
    """
    finite = (
        np.isfinite(mu) & np.isfinite(g1) & np.isfinite(g2)
        & np.isfinite(sigma1) & np.isfinite(sigma2)
    )
    with np.errstate(invalid="ignore"):
        positive = (sigma1 > 0) & (sigma2 > 0)
    bad = np.flatnonzero((np.asarray(counts) >= min_count) & ~(finite & positive))
    if bad.size == 0:
        return None
    parts = [
        f"bin {i} ({edges[i]:.4g} to {edges[i + 1]:.4g}): {int(counts[i])} galaxies, "
        f"g1={g1[i]:.3g}, g2={g2[i]:.3g}, sigma_g1={sigma1[i]:.3g}, sigma_g2={sigma2[i]:.3g}"
        for i in bad
    ]
    return (
        f"TXMeanShearSurveyProperties: WARNING - '{name}': {bad.size} bin(s) with at "
        f"least {min_count} galaxies have a non-finite mean or error, or a zero "
        "error, and are left out; check the catalog for NaN/inf shears or weights: "
        + "; ".join(parts)
    )


def _null_test_statistics(mu, g1, g2, sigma1, sigma2, counts, min_count):
    """
    Summary statistics of the binned mean shear for one survey property.

    A bin is used only if every quantity in it is finite, both errors are
    positive, and it holds at least ``min_count`` galaxies (else, poorly
    sampled bins produce incorrect error bars).

    Three statistics are returned:

    - chi2_null: the chi-squared of the mean g1 and g2 against zero shear, the
      null hypothesis, over the used bins. No parameters are fitted, so the
      number of degrees of freedom is the number of values summed (two per
      bin).
    - chi2_const: the same against a constant, each component's weighted mean
      (const_g1, const_g2), so only bin-to-bin variation counts; two fewer
      degrees of freedom, and at least two used bins are needed.
    - straight-line fits of g1 and g2 against the mean property value, whose
      slopes measure a trend with the property.

    All treat the bins as independent with shape-noise errors only; spatially
    correlated errors between bins are not included.
    """
    from scipy.stats import chi2 as chi2_dist

    usable = (
        np.isfinite(mu) & np.isfinite(g1) & np.isfinite(g2)
        & np.isfinite(sigma1) & np.isfinite(sigma2)
        & (sigma1 > 0) & (sigma2 > 0)
        & (counts >= min_count)
    )
    n = int(usable.sum())

    chi2 = float(np.sum((g1[usable] / sigma1[usable]) ** 2 + (g2[usable] / sigma2[usable]) ** 2))
    n_dof = 2 * n
    stats = {
        "chi2_null": chi2 if n_dof else np.nan,
        "n_dof_null": n_dof,
        "chi2_null_per_dof": chi2 / n_dof if n_dof else np.nan,
        "chi2_null_pvalue": float(chi2_dist.sf(chi2, n_dof)) if n_dof else np.nan,
        "n_used_bins": n,
        "min_count": min_count,
    }

    # Against a constant: subtract each component's weighted mean.
    chi2_c = 0.0
    for comp, g, s in (("g1", g1, sigma1), ("g2", g2, sigma2)):
        w = 1.0 / s[usable] ** 2
        const = float(np.sum(w * g[usable]) / np.sum(w)) if n else np.nan
        stats[f"const_{comp}"] = const
        stats[f"const_{comp}_err"] = float(1.0 / np.sqrt(np.sum(w))) if n else np.nan
        chi2_c += float(np.sum(w * (g[usable] - const) ** 2)) if n else 0.0
    n_dof_c = 2 * n - 2 if n >= 2 else 0
    stats["chi2_const"] = chi2_c if n_dof_c else np.nan
    stats["n_dof_const"] = n_dof_c
    stats["chi2_const_per_dof"] = chi2_c / n_dof_c if n_dof_c else np.nan
    stats["chi2_const_pvalue"] = float(chi2_dist.sf(chi2_c, n_dof_c)) if n_dof_c else np.nan
    for comp, g, s in (("g1", g1, sigma1), ("g2", g2, sigma2)):
        slope, slope_err, intercept, intercept_err = _weighted_line_fit(
            mu[usable], g[usable], s[usable]
        )
        stats[f"slope_{comp}"] = slope
        stats[f"slope_{comp}_err"] = slope_err
        stats[f"intercept_{comp}"] = intercept
        stats[f"intercept_{comp}_err"] = intercept_err
    return usable, stats


def _marker_offset(mu):
    """
    Plotting aid used to visually separate the g1 and g2 series in the plots.
    """
    gaps = np.diff(np.sort(mu))
    gaps = gaps[gaps > 0]
    return 0.1 * gaps.min() if gaps.size else 0.0


def _footprint_nside(map_plot_nside):
    """Resolution of the measured-galaxy footprint: the map panel's, or 1024."""
    return map_plot_nside if map_plot_nside else 1024


def _restrict_to_footprint(hsp_map, fp_nside, fp_pixels):
    """
    Copy of ``hsp_map`` with only the pixels inside the footprint (nest pixels
    at ``fp_nside`` holding measured galaxies).
    """
    import healsparse

    valid = hsp_map.valid_pixels
    nside = hsp_map.nside_sparse
    if nside >= fp_nside:
        shift = 2 * int(round(np.log2(nside // fp_nside)))
        keep = np.isin(valid >> shift, fp_pixels)
    else:
        shift = 2 * int(round(np.log2(fp_nside // nside)))
        keep = np.isin(valid, np.unique(fp_pixels >> shift))
    out = healsparse.HealSparseMap.make_empty_like(hsp_map)
    if keep.any():
        out[valid[keep]] = hsp_map[valid[keep]]
    return out


def _footprint_aspect(hsp_map):
    """
    Width over height of the map's footprint on the sky, from its coverage
    pixels. RA is measured about its circular mean, so RA = 0 wraps correctly.
    """
    import healpy as hp

    cov = np.flatnonzero(hsp_map.coverage_mask)
    if cov.size == 0:
        return 1.0
    ra, dec = hp.pix2ang(hsp_map.nside_coverage, cov, nest=True, lonlat=True)
    ra_rad = np.radians(ra)
    ra_mid = np.degrees(np.arctan2(np.sin(ra_rad).mean(), np.cos(ra_rad).mean()))
    dra = (ra - ra_mid + 180.0) % 360.0 - 180.0
    # Pad by one coverage pixel so a single-pixel field comes out square.
    pad = hp.nside2resol(hsp_map.nside_coverage, arcmin=True) / 60.0
    width = (np.ptp(dra) + pad) * np.cos(np.radians(np.median(dec)))
    height = np.ptp(dec) + pad
    return width / height


# Journal-like style for the figures, applied with rc_context so other TXPipe
# plots are not affected.
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "legend.fontsize": 9,
    "savefig.dpi": 150,
}

# Colour-blind safe colours for g1 and g2.
G1_COLOR, G2_COLOR = "#0072B2", "#D55E00"


def _plot_property_map(ax, hsp_map, nside, title, plot_nside=0, projection="McBryde"):
    """
    Draw a survey property map into ``ax`` with skyproj, as the other TXPipe
    map plots do (see ``MapsFile.plot``). skyproj zooms to the map's footprint,
    so small fields and wide or multi-field footprints are both handled.

    Maps finer than ``plot_nside`` are degraded first, which only smooths the
    displayed values. ``plot_nside=0`` keeps the native resolution.

    Returns the skyproj object (None for an empty map); the caller draws the
    colour bar once the panel has its final size.
    """
    import skyproj

    if plot_nside and nside > plot_nside:
        hsp_map = hsp_map.degrade(plot_nside, reduction="mean")

    if hsp_map.n_valid == 0:
        ax.set_title(f"{title}\n(no observed pixels)")
        ax.axis("off")
        return None

    # skyproj draws on a new axes over ``ax``, so labels go on ``sp.ax``.
    ax.axis("off")
    # Fewer grid lines, so RA labels of a small field don't overlap.
    sp = getattr(skyproj, projection + "Skyproj")(ax=ax, n_grid_lon=4, n_grid_lat=4)
    sp.draw_hspmap(hsp_map)
    # RA labels at the bottom only (as in MapsFile.plot), and smaller fonts.
    sp.ax.tick_params(axis="x", labeltop=False, labelsize=9)
    sp.ax.tick_params(axis="y", labelsize=9)
    sp.ax.set_xlabel("RA", fontsize=11)
    sp.ax.set_ylabel("Dec", fontsize=11)
    sp.ax.set_title(title, fontsize=11)
    return sp


def _projected_aspect(sp):
    """Width over height of the drawn map in projected coordinates."""
    x0, x1, y0, y1 = sp.ax.get_extent(lonlat=False)
    return abs(x1 - x0) / max(abs(y1 - y0), 1e-12)


def _center_axes_group(fig, axes, x_center):
    """
    Shift ``axes`` sideways together so they are centred on ``x_center``
    (figure fraction), measured after a draw.
    """
    fig.canvas.draw()
    boxes = [a.get_position() for a in axes]
    left = min(b.x0 for b in boxes)
    right = max(b.x1 for b in boxes)
    shift = x_center - 0.5 * (left + right)
    for a, b in zip(axes, boxes):
        a.set_position([b.x0 + shift, b.y0, b.width, b.height])


class TXMeanShearSurveyProperties(PipelineStage):
    """
    Compute the mean shear in bins of survey property maps.

    This null test checks if the mean shear correlates with spatially-varying
    survey conditions such as PSF size, depth, or sky background.

    Survey properties can come from both of:

    - A psf_maps and depth_map input, from TXPSFMaps and TXDepthMaps.
    - A directory of .hs healsparse files. Either the ``survey_property_maps``
      tag in an ingestion stage (e.g. TXIngestDataPreview1) or the
      ``external_maps_dir`` config option. Currently for DP2 we point
      external_maps_dir at the directory of .hs files.

    These map inputs are optional: alias the tags survey_property_maps, psf_maps or
    depth_map to "none" in "inputs:" in pipeline.yml to ignore any input. At
    least one map must be found, or the stage raises an error.

    For each map the output file holds the mean g1 and g2 in each property bin,
    their errors and the galaxy count, plus the chi-squared of the mean shear
    against zero and straight-line fits against the property value. Bins with
    fewer than min_count galaxies are kept in the output but left out of both
    statistics. The statistics treat bins as independent with shape-noise
    errors only.
    """

    name = "TXMeanShearSurveyProperties"
    parallel = True

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("survey_property_maps", FileCollection),
        ("psf_maps", MapsFile),
        ("depth_map", MapsFile),
    ]

    outputs = [
        ("shear_survey_property_plots", FileCollection),
        ("shear_survey_property_null", HDFFile),
    ]

    # These configs can be set in the pipeline .yml files.
    config_options = {
        "nbins": StageParameter(int, 20, msg="Number of survey property bins"),
        "chunk_rows": StageParameter(int, 100_000, msg="Rows per catalog chunk"),
        "delta_gamma": StageParameter(float, 0.02, msg="Metacal delta_gamma for shear response"),
        "outlier_fraction": StageParameter(
            float, 0.05,
            msg="Fraction of sampled galaxies excluded as outliers at each tail when computing bin edges"
        ),
        "min_count": StageParameter(
            int, 50,
            msg="Minimum number of galaxies in a bin for it to enter the chi-squared and "
                "the straight-line fits. Bins below this are still saved and plotted."
        ),
        "properties": StageParameter(
            list, [],
            msg="Map names to test (e.g. ['psf/g1_2D', 'depth/depth']). Empty list means all available."
        ),
        "psf_map_bins": StageParameter(
            list, ["2D"],
            msg="Tomographic bins of the psf_maps maps to test, matched against the "
                "trailing _<bin> of the name (e.g. psf/g1_2D). The default keeps only "
                "the non-tomographic maps: the PSF does not depend on the galaxy, so "
                "the per-bin maps sample the same field at fewer points. Empty list "
                "means all bins."
        ),
        "external_maps_dir": StageParameter(
            str, "",
            msg="Directory (or filename prefix) of .hs healsparse files to test. "
                "Only used when the survey_property_maps input is not supplied."
        ),
        "bands": StageParameter(
            list, ["i"],
            msg="Bands to keep when loading .hs files, matched against the "
                "trailing _<band> in the filename. Empty list means all bands. "
                "Files with no band suffix are always kept."
        ),
        "map_plot_nside": StageParameter(
            int, 256,
            msg="Nside to degrade survey property maps to for the map panel "
                "(0 to plot at the map's native resolution)"
        ),
        "projection": StageParameter(
            str, "McBryde",
            msg="Skyproj projection for the map panel (e.g. McBryde, Mollweide)"
        ),
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        import matplotlib
        matplotlib.use("Agg")

        # Pull corresponding shear columns for this catalog.
        cols = self.catalog_columns()

        # Load survey property maps.
        all_maps = self.load_maps()

        # Choose bin edges based on range of *galaxy* survey property values.
        bin_edges, counts = self.choose_bin_edges(all_maps, cols)

        # Calculate mean shear and uncertainty for these bin edges.
        binned_shears = self.measure_mean_shear(all_maps, bin_edges, cols)

        # Compute statistics, and save results to files/plots.
        self.write_results(all_maps, binned_shears, bin_edges, counts)

    def catalog_columns(self):
        """
        Work out which shear and tomography columns to read, using the shear
        catalog class's own methods.

        Returns a dict with the catalog type, the column lists, the keys of the
        positions the property is looked up at, and the source-bin column.
        """
        cat_type = read_shear_catalog_type(self)

        with self.open_input("shear_catalog", wrapper=True) as f:
            prefix = f.get_primary_catalog_prefix()
            tomo_cols = list(f.get_tomography_bin_columns())
            source_bin_key = f.get_primary_tomography_bin_column()
            if prefix:
                # Metadetect types: every variant is its own detection catalog,
                # so positions and weights are read per variant too.
                shear_cols = list(f.get_column_name_variants("ra", "dec", "g1", "g2", "weight"))
            else:
                # Metacal, anacal, lensfit, hsc: one set of positions.
                shear_cols = list(f.get_column_name_variants("g1", "g2"))
                shear_cols += ["ra", "dec", "weight"]
                shear_cols += f.get_extra_calibration_columns(mode="compute")

        return {
            "cat_type": cat_type,
            # Drop duplicate shear_cols
            "shear_cols": list(dict.fromkeys(shear_cols)),
            "tomo_cols": tomo_cols,
            "ra_key": f"{prefix}ra",
            "dec_key": f"{prefix}dec",
            "source_bin_key": source_bin_key,
        }

    def load_maps(self):
        """
        Load every survey property map for the run: the psf_maps and depth_map
        inputs, then a directory of .hs files. Returns a dict of
        name -> (healsparse map, nside), filtered by the properties option.

        Raises an error if no map is found.
        """
        import healsparse

        properties_filter = self.config["properties"]
        all_maps = {}

        def register(name, hsp_map, nside):
            if properties_filter and name not in properties_filter:
                return
            all_maps[name] = (hsp_map, nside)

        # Aliased to "none" these are skipped, letting the test run without the
        # map stages having been run first.
        for tag in ("psf_maps", "depth_map"):
            if self.get_input(tag) == "none":
                continue
            with self.open_input(tag, wrapper=True) as f:
                for name in f.list_maps():
                    if tag == "psf_maps" and not _keep_psf_map(name, self.config["psf_map_bins"]):
                        continue
                    register(name, f.read_map(name), f.read_map_info(name)["nside"])

        map_dir = self.survey_property_map_dir()
        paths = _hs_files(map_dir)
        bands = self.config["bands"]
        # A file with no band suffix is kept whatever bands is set to, so
        # this cannot silently drop maps that are not named the DP1 way.
        paths = [
            p for p in paths
            if not bands or _band_of(pathlib.Path(p).stem) in (None, *bands)
        ]
        if paths and self.rank == 0:
            print(f"TXMeanShearSurveyProperties: {len(paths)} .hs maps from {map_dir}")
        names = _hs_map_names(paths)
        for path in paths:
            hsp_map = healsparse.HealSparseMap.read(path)
            register(names[path], hsp_map, hsp_map.nside_sparse)

        if not all_maps:
            raise ValueError(
                "TXMeanShearSurveyProperties: no survey property maps found. Supply "
                "psf_maps, depth_map, survey_property_maps or external_maps_dir, and "
                "check the properties and bands options."
            )
        return all_maps

    def survey_property_map_dir(self):
        """
        The directory of .hs files to load. The survey_property_maps input takes
        precedence over the external_maps_dir option.
        """
        map_dir = self.get_input("survey_property_maps")
        if map_dir == "none":
            return self.config["external_maps_dir"]
        if not _hs_files(map_dir):
            if self.rank == 0:
                print(
                    "TXMeanShearSurveyProperties: no .hs files in the "
                    f"survey_property_maps input at {map_dir}."
                )
            return self.config["external_maps_dir"]
        return map_dir

    def catalog_iterator(self, cols):
        return self.combined_iterators(
            self.config["chunk_rows"],
            "shear_catalog", "shear", cols["shear_cols"],
            "shear_tomography_catalog", "tomography", cols["tomo_cols"],
            longest=True,
        )

    def choose_bin_edges(self, all_maps, cols):
        """
        Pass 1 over the catalog: choose bin edges for each map from the property
        values at the galaxies that actually enter the mean shear measurement.

        The same values give the number of galaxies in each bin, used for the
        min_count cut. Maps with no usable range are removed from all_maps.

        Returns dicts of name -> bin edges and name -> counts per bin.
        """
        nbins = self.config["nbins"]
        outlier_fraction = self.config["outlier_fraction"]
        ra_key, dec_key = cols["ra_key"], cols["dec_key"]

        import healpy as hp

        # Record the pixels with measured galaxies, so the map panel shows
        # only the region the statistics come from.
        fp_nside = _footprint_nside(self.config["map_plot_nside"])
        fp_pixels = []
        sampled = {name: [] for name in all_maps}
        for s, e, data in self.catalog_iterator(cols):
            if self.rank == 0:
                print(f"TXMeanShearSurveyProperties: bin-edge pass rows {s:,} – {e:,}")
            in_source_bin = data[cols["source_bin_key"]] != -1
            fp_pixels.append(np.unique(hp.ang2pix(
                fp_nside, data[ra_key][in_source_bin], data[dec_key][in_source_bin],
                lonlat=True, nest=True,
            )))
            for name, (hsp_map, nside) in all_maps.items():
                vals = _property_values_at(hsp_map, nside, data[ra_key], data[dec_key])
                sampled[name].append(vals[in_source_bin & np.isfinite(vals)])

        fp_pixels = np.unique(np.concatenate(fp_pixels)) if fp_pixels else np.empty(0, int)
        if self.comm is not None:
            fp_pixels = np.unique(np.concatenate(self.comm.allgather(fp_pixels)))
        self.galaxy_footprint = (fp_nside, fp_pixels)

        # Combine the sampled values across chunks and MPI ranks. allgather so
        # every rank builds identical binners later.
        bin_edges, counts = {}, {}
        for name in list(all_maps):
            local = np.concatenate(sampled.pop(name)) if sampled[name] else np.empty(0)
            if self.comm is not None:
                local = np.concatenate(self.comm.allgather(local))
            edges = _bin_edges_from_values(local, nbins, outlier_fraction)
            if edges is None:
                if self.rank == 0:
                    print(
                        f"TXMeanShearSurveyProperties: skipping '{name}' (no "
                        "usable range among the galaxies that sample it)."
                    )
                del all_maps[name]
                continue
            bin_edges[name] = edges
            counts[name] = _counts_in_bins(local, edges)

        if not all_maps:
            raise ValueError(
                "TXMeanShearSurveyProperties: no survey property map has a usable "
                "range among the measured galaxies."
            )
        return bin_edges, counts

    def measure_mean_shear(self, all_maps, bin_edges, cols):
        """
        Pass 2 over the catalog: accumulate the calibrated mean shear in the
        property bins of every map, in one sweep. Returns name -> MeanShearInBins.
        """
        cat_type = cols["cat_type"]
        binned_shears = {
            name: MeanShearInBins(
                "survey_prop",
                bin_edges[name],
                self.config["delta_gamma"],
                cut_source_bin=True,
                shear_catalog_type=cat_type,
            )
            for name in all_maps
        }

        for s, e, data in self.catalog_iterator(cols):
            if self.rank == 0:
                print(f"TXMeanShearSurveyProperties: measurement pass rows {s:,} – {e:,}")
            for name, (hsp_map, nside) in all_maps.items():
                # NaN for galaxies on unobserved pixels, which the binner drops.
                # For metadetect this writes one column per shear variant.
                _assign_property_column(
                    data, cat_type, hsp_map, nside, cols["ra_key"], cols["dec_key"]
                )
                binned_shears[name].add_data(data)
        return binned_shears

    def write_results(self, all_maps, binned_shears, bin_edges, counts):
        """
        Collect each map's binned mean shear, then on rank 0 write it with its
        statistics to the HDF5 output and make its plot.

        The loop runs on every rank because binner.collect is collective and
        would deadlock if some ranks skipped it.
        """
        min_count = self.config["min_count"]

        # Prevent race, only rank 0 writes
        if self.rank == 0:
            output_dir = self.open_output("shear_survey_property_plots", wrapper=True)
            hdf_out = self.open_output("shear_survey_property_null")

        png_files = []
        for name, binner in binned_shears.items():
            # MeanShearInBins.collect gathers all partial sums and sends to all ranks,
            # so all ranks must call it.
            mu, g1, g2, sigma1, sigma2 = binner.collect(self.comm)
            if self.rank != 0:
                continue

            safe_name = name.replace("/", "_")
            usable, stats = _null_test_statistics(
                mu, g1, g2, sigma1, sigma2, counts[name], min_count
            )
            cut_report = _min_count_cut_report(name, bin_edges[name], counts[name], min_count)
            if cut_report:
                print(cut_report)
            bad_report = _bad_bin_report(
                name, bin_edges[name], mu, g1, g2, sigma1, sigma2, counts[name], min_count
            )
            if bad_report:
                print(bad_report)
            if stats["n_used_bins"] < _MIN_FIT_BINS:
                print(
                    f"TXMeanShearSurveyProperties: WARNING - '{name}' has only "
                    f"{stats['n_used_bins']} bins with at least {min_count} galaxies; "
                    "its slopes are written as NaN."
                )

            grp = hdf_out.create_group(safe_name)
            for key, value in (
                ("bin_centers", mu), ("g1", g1), ("g2", g2),
                ("sigma_g1", sigma1), ("sigma_g2", sigma2),
                ("count", counts[name]), ("used_in_statistics", usable),
            ):
                grp.create_dataset(key, data=value)
            for key, value in stats.items():
                grp.attrs[key] = value

            png_name = f"{safe_name}.png"
            self.plot_property(
                output_dir.path_for_file(png_name), name, all_maps[name],
                mu, g1, g2, sigma1, sigma2, usable, stats,
            )
            png_files.append(png_name)

        # Prevent race, only rank 0 writes
        if self.rank == 0:
            output_dir.write_listing(png_files)
            hdf_out.close()

    def plot_property(self, path, name, map_info, mu, g1, g2, sigma1, sigma2, usable, stats):
        """
        Two-panel figure: the property map on the left, so the trend on the right
        can be read against the spatial structure driving it.

        Bins left out of the statistics (too few galaxies) are drawn as open
        grey markers so they stay visible without looking like part of the fit.
        """
        import matplotlib.pyplot as plt

        with plt.rc_context(PAPER_STYLE):
            fig = self._property_figure(name, map_info, mu, g1, g2, sigma1, sigma2,
                                        usable, stats)
            fig.savefig(path)
        plt.close(fig)

    def _property_figure(self, name, map_info, mu, g1, g2, sigma1, sigma2, usable, stats):
        import matplotlib.pyplot as plt

        safe_name = name.replace("/", "_")
        hsp_map, map_nside = map_info
        footprint = getattr(self, "galaxy_footprint", None)
        if footprint is not None:
            hsp_map = _restrict_to_footprint(hsp_map, *footprint)
        # A very wide footprint goes above the trend panel instead of beside it.
        wide = _footprint_aspect(hsp_map) > 2.5
        # Lengths in inches.
        width = 10.0 if wide else 12.0
        left, right = 0.85, 0.35
        if wide:
            # The map row height is set below, from the drawn map's shape.
            top, gap, bottom, trend_h, map_h = 0.4, 0.95, 0.9, 3.6, 2.0
            height = top + map_h + gap + trend_h + bottom
            fig = plt.figure(figsize=(width, height))
            gs = fig.add_gridspec(2, 1, height_ratios=[map_h, trend_h])
        else:
            height = 4.8
            fig = plt.figure(figsize=(width, height))
            # The map panel is a touch narrower: a sky patch keeps its aspect
            # ratio, so given equal widths it would sit small beside the trend.
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.2,
                                  left=0.05, right=0.97, top=0.86, bottom=0.2)

        map_ax = fig.add_subplot(gs[0])
        sp = _plot_property_map(
            map_ax, hsp_map, map_nside, safe_name,
            plot_nside=self.config["map_plot_nside"],
            projection=self.config["projection"],
        )
        cbar_w = 0.45  # colour bar and its tick labels, beside a wide map
        if wide:
            # Make the map row as tall as the map is at full width.
            map_w = width - left - right - cbar_w
            if sp is not None:
                map_h = float(np.clip(map_w / _projected_aspect(sp), 1.0, 5.0))
            height = top + map_h + gap + trend_h + bottom
            fig.set_size_inches(width, height)
            gs.set_height_ratios([map_h, trend_h])
            gs.update(left=left / width, right=1 - right / width,
                      top=1 - top / height, bottom=bottom / height,
                      hspace=gap / ((map_h + trend_h) / 2))
        ax = fig.add_subplot(gs[1])

        if sp is not None:
            # Move skyproj's axes to the placeholder's final position.
            box = map_ax.get_position()
            if wide:
                box.x1 -= cbar_w / width
            sp.ax.set_position(box)
            # Vertical colour bar beside a wide map, horizontal under a compact one.
            if wide:
                cbar = sp.draw_colorbar(location="right", fraction=0.025, pad=0.015,
                                        fontsize=9)
            else:
                cbar = sp.draw_colorbar(label=safe_name, location="bottom",
                                        fraction=0.05, pad=0.16, fontsize=9)
            cbar.outline.set_linewidth(0.6)
            # A few ticks, with an offset for tiny ranges, so labels don't overlap.
            from matplotlib.ticker import MaxNLocator, ScalarFormatter
            cbar.locator = MaxNLocator(nbins=6 if wide else 5)
            cbar.formatter = ScalarFormatter(useOffset=True)
            cbar.formatter.set_powerlimits((-3, 4))
            cbar.update_ticks()
            # Centre the map and colour bar (over the trend panel when stacked).
            cell = map_ax.get_position()
            centre = ax.get_position() if wide else cell
            _center_axes_group(fig, [sp.ax, cbar.ax], 0.5 * (centre.x0 + centre.x1))

        finite = np.isfinite(mu) & np.isfinite(g1) & np.isfinite(g2)
        # Small horizontal offset so the g1/g2 series don't overlap, taken over
        # the finite bins only. g1 sits to the right of each bin and g2 to the
        # left, matching TXSourceDiagnosticPlots.
        dx = _marker_offset(mu[finite])
        ax.axhline(0, color="0.3", lw=0.7, ls="--", zorder=0)

        for comp, g, s, sign, marker, color in (
            ("g1", g1, sigma1, +1, "s", G1_COLOR),
            ("g2", g2, sigma2, -1, "o", G2_COLOR),
        ):
            x = mu + sign * dx
            used = usable
            excluded = finite & ~usable
            slope, slope_err = stats[f"slope_{comp}"], stats[f"slope_{comp}_err"]
            label = (
                f"$g_{comp[1]}$  ($m$ = {slope:.2e} $\\pm$ {slope_err:.2e})"
                if np.isfinite(slope) else f"$g_{comp[1]}$  (too few bins to fit)"
            )
            ax.errorbar(x[used], g[used], s[used], fmt=marker, markersize=4,
                        color=color, mec=color, elinewidth=0.8, capsize=0,
                        zorder=2, label=label)
            if excluded.any():
                ax.errorbar(x[excluded], g[excluded], s[excluded], fmt=marker,
                            markersize=4, mfc="none", color="0.6", elinewidth=0.6,
                            zorder=1)
            # The fitted line, drawn across the range that was actually fitted.
            if np.isfinite(slope):
                x_line = np.array([mu[used].min(), mu[used].max()])
                ax.plot(x_line, slope * x_line + stats[f"intercept_{comp}"],
                        color=color, lw=1.1, zorder=3)

        if (finite & ~usable).any():
            # Drawn as an errorbar so the legend lists it after g1 and g2.
            ax.errorbar([], [], [], fmt="o", markersize=4, mfc="none", color="0.6",
                        elinewidth=0.6, label=f"< {stats['min_count']} galaxies (not used)")

        ax.set_xlabel(safe_name)
        ax.set_ylabel("Mean shear")
        # Against zero and against a constant, to tell an offset from a trend.
        ax.set_title(
            f"vs zero: $\\chi^2$/dof = {stats['chi2_null']:.1f}/{stats['n_dof_null']}, "
            f"$p$ = {stats['chi2_null_pvalue']:.2g}\n"
            f"vs constant: $\\chi^2$/dof = {stats['chi2_const']:.1f}/{stats['n_dof_const']}, "
            f"$p$ = {stats['chi2_const_pvalue']:.2g}",
        )
        # Legend below the panel, in fewer columns if one row doesn't fit.
        fig.canvas.draw()
        for ncol in (3, 2, 1):
            leg = ax.legend(loc="upper center",
                            bbox_to_anchor=(0.5, -0.13 if wide else -0.14),
                            ncol=ncol, frameon=False, columnspacing=1.0,
                            handletextpad=0.3)
            box = leg.get_window_extent(fig.canvas.get_renderer())
            if box.x0 >= 0 and box.x1 <= fig.bbox.width:
                break
        return fig

