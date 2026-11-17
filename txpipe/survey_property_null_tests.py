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
# META_VARIANTS is not used here, but is re-exported for the tests.
from .calibration.calibration_utils.names import META_VARIANTS
from .utils import read_shear_catalog_type

BANDS = "ugrizy"

# Fewest usable bins a straight-line fit is attempted on. Two points always fit
# a line exactly and leave no freedom to judge it, so at least three are needed.
_MIN_FIT_BINS = 3


def _hs_files(root):
    """
    List the .hs healsparse files under a survey property map root.

    ``root`` may be a directory, e.g. what the survey_property_maps FileCollection
    from ingestion stage, or a filename prefix.
    """
    if not root:
        return []
    pattern = f"{root.rstrip('/')}/*.hs" if os.path.isdir(root) else f"{root}*.hs"
    return sorted(glob.glob(pattern))


def _band_of(stem):
    """
    The band a survey property map ends in, or None if no band
    suffix. DP1 writes one file per band, named e.g. ..._weighted_mean_i.
    """
    band = stem.rpartition("_")[2]
    return band if band in BANDS and len(band) == 1 else None


def _shorten_map_name(stem):
    """
    Trim a survey property filename down for plotting.

    E.g. replaces ``deepCoadd_<property>_consolidated_map_<reduction>_<band>``
    with <property>_<reduction>_<band>, such as epoch_max_i.

    The <reduction> is kept when it is not weighted_mean.

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
    Choose the name to register each .hs file under.
    """
    stems = {path: pathlib.Path(path).stem for path in paths}
    short = {path: _shorten_map_name(stem) for path, stem in stems.items()}
    counts = collections.Counter(short.values())
    return {
        path: (stems[path] if counts[name] > 1 else name)
        for path, name in short.items()
    }


def _bin_edges_from_values(vals, nbins, outlier_fraction):
    """
    Compute bin edges for a survey property from the values it takes within the
    shear catalog footprint.

    ``vals`` is the survey property sampled at ``{galaxies with defined shear
    in a source bin} ∩ {pixels where that map is observed}`` -- exactly the sample
    the mean shear is measured over -- so the bins span the region actually
    being tested, with no separate mask needed. Non-finite entries must already
    be removed by the caller.

    Maps taking only a few distinct values (integer counts such as
    ``bright_objects/count``, or flag maps) are binned with one bin per value,
    centred on that value. Evenly spaced edges would instead land exactly *on*
    the integers, and since ``MeanShearInBins.selector`` selects with strict
    inequalities on both sides, every galaxy would then fall into no bin at all
    and the test would silently come out empty.

    Returns ``None`` if the sample has no usable range -- because no galaxy
    sampled the map, or because the property is constant across the sample and
    so carries no trend to test. That map is then skipped.
    """
    if vals.size == 0:
        return None

    lo = np.percentile(vals, 100 * outlier_fraction)
    hi = np.percentile(vals, 100 * (1 - outlier_fraction))

    # Discrete-valued map: put the edges between the values so each value sits
    # at a bin centre. Checked on the values that survive the outlier trim, and
    # on their number rather than the dtype, so that float maps which happen to
    # be discrete are handled too.
    uniq = np.unique(vals[(vals >= lo) & (vals <= hi)])
    if uniq.size < 2:
        # Constant across the sample: no trend can be measured.
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

    return np.linspace(lo, hi, nbins + 1)


def _counts_in_bins(vals, edges):
    """
    Number of values in each bin, selected the way ``MeanShearInBins.selector``
    selects them: strictly inside both edges.

    ``np.histogram`` is not used because it includes the lower edge of each bin
    and the upper edge of the last one, so its counts would not match the
    galaxies the mean shear is actually computed from.
    """
    return np.array(
        [np.count_nonzero((vals > lo) & (vals < hi)) for lo, hi in zip(edges[:-1], edges[1:])],
        dtype=np.int64,
    )


def _property_values_at(hsp_map, nside, ra, dec):
    """
    Look up a survey property map at galaxy positions, returning float values
    with ``NaN`` for galaxies on pixels the map does not observe.

    Unobserved pixels are detected with HealSparse's own validity mask rather
    than by comparing against ``hp.UNSEEN``: the unobserved sentinel differs by
    dtype (``hp.UNSEEN`` for float64, but a distinct value for integer and
    float32 maps), so an equality test would silently miss them. ``NaN`` values
    fall outside every bin and are dropped by ``MeanShearInBins``, so each
    property is tested only where it is actually observed.
    """
    import healpy as hp

    pixels = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    # valid_mask=True flags observed pixels for any map dtype.
    valid = hsp_map.get_values_pix(pixels, valid_mask=True)
    prop_vals = hsp_map[pixels].astype(float)
    prop_vals[~valid] = np.nan
    return prop_vals


def _detection_variants(cat_type):
    """
    The shear-variant group names (ns, 1p, ...) for catalog types where each
    variant is a separate detection catalog with its own positions (metadetect
    and scalar metadetect), or None for types whose variants share one set of
    positions (metacal, anacal, lensfit, hsc).

    Taken from the shear catalog class, so new catalog types are handled by
    their own class rather than by a list kept here.
    """
    cat_class = class_for_shear_catalog_type(cat_type)
    if not cat_class.get_primary_catalog_prefix():
        return None
    return [p.rstrip("/") for p in cat_class.get_classifier_prefixes()]


def _assign_property_column(data, cat_type, hsp_map, nside, ra_key, dec_key):
    """
    Set the ``survey_prop`` column(s) on a data chunk for ``MeanShearInBins``.

    For metadetect each shear variant (ns/1p/1m/2p/2m) is a *separate* detection
    catalogue with its own positions and its own length, and the calibrator
    evaluates the bin selector once per variant. A single property column sized
    to the unsheared variant therefore cannot be broadcast against the per-
    variant ``1p/bin`` etc. columns. So one ``{variant}/survey_prop`` column is
    written per variant, each looked up at that variant's own positions, matched
    by ``_DataWrapper`` in the calibrator. ``MeanShearInBins.add_data`` also
    reads a plain ``survey_prop`` indexed by the *unsheared* selection when it
    accumulates the mean property value, so that is mirrored from the unsheared
    variant.

    Other catalogue types (metacal, lensfit, hsc) share one position array and
    length across variants, so a single ``survey_prop`` column suffices.
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
    Weighted least-squares fit of ``y = slope * x + intercept``.

    The errors on the parameters come from ``sigma`` alone, treated as the true
    one-sigma errors on ``y``. They are not rescaled by the scatter of the
    points about the line, which ``scipy.optimize.curve_fit`` does by default
    (and so ``txpipe.utils.fitting.fit_straight_line``): with that rescaling, a
    set of points that happens to lie closer to a line than its error bars
    suggest gets a smaller slope error than the data support.

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


def _null_test_statistics(mu, g1, g2, sigma1, sigma2, counts, min_count):
    """
    Summary statistics of the binned mean shear for one survey property.

    A bin is used only if every quantity in it is finite, both errors are
    positive, and it holds at least ``min_count`` galaxies. The count cut
    matters because each bin's error is the spread of its galaxies' shears
    divided by sqrt(N_eff): with only a few galaxies the spread is itself badly
    measured and can come out far too small, giving that bin a huge weight in
    both statistics below.

    Two statistics are returned:

    - chi2_null: the chi-squared of the mean g1 and g2 against zero shear, the
      null hypothesis, over the used bins. No parameters are fitted, so the
      number of degrees of freedom is the number of values summed (two per
      bin).
    - straight-line fits of g1 and g2 against the mean property value, whose
      slopes measure a trend with the property.

    Both treat the bins as independent with shape-noise errors only; spatially
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
    Horizontal offset used to separate the g1 and g2 series in the plots.

    ``mu`` holds the *mean* property value of the galaxies in each populated
    bin, not the bin midpoint, so the points are not evenly spaced. The offset
    is therefore scaled from the narrowest gap rather than a typical one: an
    offset wider than half a gap would push a point past its neighbour and show
    the series in the wrong order.

    Empty bins (``NaN`` from ``collect``) must be removed before calling, or the
    spacing is undefined. Returns 0 when there is no spacing to work with -- a
    single point, or several points sharing one value -- in which case the two
    series simply sit on top of each other.
    """
    gaps = np.diff(np.sort(mu))
    gaps = gaps[gaps > 0]
    return 0.1 * gaps.min() if gaps.size else 0.0


def _plot_property_map(ax, hsp_map, nside, title, plot_nside=0, projection="McBryde"):
    """
    Draw a survey property map into ``ax`` with skyproj, as the other TXPipe
    map plots do (see ``MapsFile.plot``). skyproj zooms to the map's footprint,
    so small fields and wide or multi-field footprints are both handled.

    Maps finer than ``plot_nside`` are degraded first, which only smooths the
    displayed values. ``plot_nside=0`` keeps the native resolution.
    """
    import skyproj

    if plot_nside and nside > plot_nside:
        hsp_map = hsp_map.degrade(plot_nside, reduction="mean")

    if hsp_map.n_valid == 0:
        ax.set_title(f"{title}\n(no observed pixels)")
        ax.axis("off")
        return

    sp = getattr(skyproj, projection + "Skyproj")(ax=ax)
    sp.draw_hspmap(hsp_map)
    sp.draw_colorbar()
    ax.set_title(title)


class TXMeanShearSurveyProperties(PipelineStage):
    """
    Compute the mean shear in bins of survey property maps.

    This null test checks if the mean shear correlates with spatially-varying
    survey conditions such as PSF size, depth, or sky background.

    Survey properties can come from both of:

    - A psf_maps and depth_map input, from TXPSFMaps and TXDepthMaps.
    - A directory of .hs healsparse files. This is the survey_property_maps
      FileCollection written by TXIngestDataPreview1 when that input is wired
      up, and otherwise the directory named by the external_maps_dir config
      option. The DP2 ingestion stages do not yet write survey property maps,
      so for DP2, and for SUPREME-style or other external maps, point
      external_maps_dir at the directory of .hs files.

    These map inputs are optional: alias the tags survey_property_maps, psf_maps or
    depth_map to "none" in "inputs:" in pipeline.yml to ignore any input. At
    least one map must be found, or the stage raises an error.

    The bands config option filters the .hs files down to chosen bands, since
    DP1 ingestion writes every property in all six of ugrizy. Files with no
    recognised band suffix are always kept.

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
        import matplotlib
        matplotlib.use("Agg")

        cols = self.catalog_columns()
        all_maps = self.load_maps()
        bin_edges, counts = self.choose_bin_edges(all_maps, cols)
        binned_shears = self.measure_mean_shear(all_maps, bin_edges, cols)
        self.write_results(all_maps, binned_shears, counts)

    def catalog_columns(self):
        """
        Work out which shear and tomography columns to read, using the shear
        catalog class's own methods so each catalog type defines its columns in
        one place.

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
                # so positions and weights are read per variant too (see
                # _assign_property_column).
                shear_cols = list(f.get_column_name_variants("ra", "dec", "g1", "g2", "weight"))
            else:
                # Metacal, anacal, lensfit, hsc: one set of positions. The
                # sheared columns (metacal, anacal) come from the variant names,
                # and the response or m is recomputed per property bin from them,
                # since a re-binned selection has a different response from the
                # one saved for each tomographic bin.
                shear_cols = list(f.get_column_name_variants("g1", "g2"))
                shear_cols += ["ra", "dec", "weight"]
                shear_cols += f.get_extra_calibration_columns(mode="compute")

        return {
            "cat_type": cat_type,
            # dict.fromkeys drops duplicates (e.g. lensfit's extra "dec") in order.
            "shear_cols": list(dict.fromkeys(shear_cols)),
            "tomo_cols": tomo_cols,
            "ra_key": f"{prefix}ra",
            "dec_key": f"{prefix}dec",
            "source_bin_key": source_bin_key,
        }

    def load_maps(self):
        """
        Load every survey property map to test: the psf_maps and depth_map
        inputs, then a directory of .hs files. Returns a dict of
        name -> (healsparse map, nside), filtered by the properties option.

        Raises an error if no map is found, since an empty test would otherwise
        pass silently.
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
        precedence over the external_maps_dir option. If the input is wired up
        but holds no .hs files, fall back to the option rather than silently
        testing nothing.
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
        values at the galaxies that actually enter the measurement, i.e.
        galaxies in a source bin where the map is observed. This is the same
        region the mean shear is measured over, so the bins match the tested
        region without needing a separate mask.

        The same values give the number of galaxies in each bin, used for the
        min_count cut. Maps with no usable range are removed from all_maps.

        Returns dicts of name -> bin edges and name -> counts per bin.
        """
        nbins = self.config["nbins"]
        outlier_fraction = self.config["outlier_fraction"]
        ra_key, dec_key = cols["ra_key"], cols["dec_key"]

        sampled = {name: [] for name in all_maps}
        for s, e, data in self.catalog_iterator(cols):
            if self.rank == 0:
                print(f"TXMeanShearSurveyProperties: bin-edge pass rows {s:,} – {e:,}")
            in_source_bin = data[cols["source_bin_key"]] != -1
            for name, (hsp_map, nside) in all_maps.items():
                vals = _property_values_at(hsp_map, nside, data[ra_key], data[dec_key])
                sampled[name].append(vals[in_source_bin & np.isfinite(vals)])

        # Combine the sampled values across chunks and MPI ranks. allgather so
        # every rank builds identical binners later (their construction and
        # collection are collective through the calibrators).
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

    def write_results(self, all_maps, binned_shears, counts):
        """
        Collect each map's binned mean shear, then on rank 0 write it with its
        statistics to the HDF5 output and make its plot.

        The loop runs on every rank because binner.collect is collective and
        would deadlock if some ranks skipped it. Only rank 0 opens the outputs,
        since every rank opening the same file would race.
        """
        min_count = self.config["min_count"]
        if self.rank == 0:
            output_dir = self.open_output("shear_survey_property_plots", wrapper=True)
            hdf_out = self.open_output("shear_survey_property_null")

        png_files = []
        for name, binner in binned_shears.items():
            mu, g1, g2, sigma1, sigma2 = binner.collect(self.comm)
            if self.rank != 0:
                continue

            safe_name = name.replace("/", "_")
            usable, stats = _null_test_statistics(
                mu, g1, g2, sigma1, sigma2, counts[name], min_count
            )
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

        safe_name = name.replace("/", "_")
        fig = plt.figure(figsize=(12.0, 4.5))
        # The map panel is a touch narrower: a sky patch keeps its aspect
        # ratio, so given equal widths it would sit small beside the trend.
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])

        hsp_map, map_nside = map_info
        _plot_property_map(
            fig.add_subplot(gs[0]), hsp_map, map_nside, safe_name,
            plot_nside=self.config["map_plot_nside"],
            projection=self.config["projection"],
        )

        ax = fig.add_subplot(gs[1])
        finite = np.isfinite(mu) & np.isfinite(g1) & np.isfinite(g2)
        # Small horizontal offset so the g1/g2 series don't overlap, taken over
        # the finite bins only. g1 sits to the right of each bin and g2 to the
        # left, matching TXSourceDiagnosticPlots.
        dx = _marker_offset(mu[finite])
        ax.axhline(0, color="k", lw=0.8, ls="--")

        for comp, g, s, sign, marker, color in (
            ("g1", g1, sigma1, +1, "s", "tab:blue"),
            ("g2", g2, sigma2, -1, "o", "tab:orange"),
        ):
            x = mu + sign * dx
            used = usable
            excluded = finite & ~usable
            # Error bars are drawn in black on top of the markers, so that a bin
            # with a tiny error still shows a short line through its marker.
            ax.errorbar(x[used], g[used], s[used], fmt="none", ecolor="k",
                        elinewidth=0.8, zorder=3)
            slope, slope_err = stats[f"slope_{comp}"], stats[f"slope_{comp}_err"]
            label = (
                f"{comp}  (m={slope:.2e} $\\pm$ {slope_err:.2e})"
                if np.isfinite(slope) else f"{comp}  (too few bins to fit)"
            )
            ax.plot(x[used], g[used], marker, markersize=4, color=color,
                    linestyle="none", zorder=2, label=label)
            if excluded.any():
                ax.errorbar(x[excluded], g[excluded], s[excluded], fmt=marker,
                            markersize=4, mfc="none", color="0.6", elinewidth=0.6,
                            zorder=1)
            # The fitted line, drawn across the range that was actually fitted.
            if np.isfinite(slope):
                x_line = np.array([mu[used].min(), mu[used].max()])
                ax.plot(x_line, slope * x_line + stats[f"intercept_{comp}"],
                        color=color, lw=1)

        if (finite & ~usable).any():
            ax.plot([], [], "o", mfc="none", color="0.6", linestyle="none",
                    label=f"< {stats['min_count']} galaxies (not used)")

        ax.set_xlabel(safe_name)
        ax.set_ylabel("Mean shear")
        ax.set_title(
            f"{safe_name}  ($\\chi^2_{{\\rm null}}$/dof = "
            f"{stats['chi2_null']:.1f}/{stats['n_dof_null']}, "
            f"p = {stats['chi2_null_pvalue']:.2g})"
        )
        ax.legend(fontsize=8)
        # Explicit spacing rather than tight_layout: the map colorbar and the
        # trend panel need a wide gap between them.
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.15, wspace=0.32)
        fig.savefig(path)
        plt.close(fig)
