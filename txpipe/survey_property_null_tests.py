import glob
import pathlib
import numpy as np
from ceci.config import StageParameter

from .base_stage import PipelineStage
from .data_types import ShearCatalog, TomographyCatalog, MapsFile, FileCollection, HDFFile
from .shear_calibration import MeanShearInBins, metadetect_variants, META_VARIANTS
from .utils import read_shear_catalog_type
from .utils.fitting import fit_straight_line, calc_chi2
from .diagnostics import where_all_finite


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


def _assign_property_column(data, cat_type, hsp_map, nside, ra_key, dec_key):
    """
    Set the ``survey_prop`` column(s) on a data chunk for ``MeanShearInBins``.

    For metadetect each shear variant (ns/1p/1m/2p/2m) is a *separate* detection
    catalogue with its own positions and its own length, and the calibrator
    evaluates the bin selector once per variant. A single property column sized
    to the unsheared variant therefore cannot be broadcast against the per-
    variant ``bin_1p`` etc. columns. So one ``{variant}/survey_prop`` column is
    written per variant, each looked up at that variant's own positions, matched
    by ``_DataWrapper`` in the calibrator. ``MeanShearInBins.add_data`` also
    reads a plain ``survey_prop`` indexed by the *unsheared* selection when it
    accumulates the mean property value, so that is mirrored from the unsheared
    variant.

    Other catalogue types (metacal, lensfit, hsc) share one position array and
    length across variants, so a single ``survey_prop`` column suffices.
    """
    if cat_type == "metadetect":
        for v in META_VARIANTS:
            data[f"{v}/survey_prop"] = _property_values_at(
                hsp_map, nside, data[f"{v}/ra"], data[f"{v}/dec"]
            )
        data["survey_prop"] = data[f"{META_VARIANTS[0]}/survey_prop"]
    else:
        data["survey_prop"] = _property_values_at(
            hsp_map, nside, data[ra_key], data[dec_key]
        )


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


# Above this angular span (degrees, in RA or Dec) a footprint no longer fits a
# zoomed flat panel -- e.g. several DP1 fields scattered across the sky -- and
# an all-sky projection reads better than two specks in a sea of white.
_WIDE_FOOTPRINT_DEG = 90.0


def _footprint_ra_span(ra):
    """
    Angular RA extent of a footprint, accounting for the 0/360 wrap.

    The footprint occupies the complement of its single largest empty RA gap,
    so the true span is 360 minus that gap. Computing it this way is correct
    whether or not the footprint straddles RA=0, where a naive ``max - min``
    would wrongly report almost the whole sky.
    """
    if ra.size < 2:
        return 0.0
    ra_sorted = np.sort(ra)
    interior_gap = np.diff(ra_sorted).max()
    wrap_gap = 360.0 - (ra_sorted[-1] - ra_sorted[0])
    return 360.0 - max(interior_gap, wrap_gap)


def _plot_property_map(ax, fig, hsp_map, nside, title, plot_nside=0, width=600):
    """
    Draw a survey property map into ``ax`` with a white, gridded background.

    All observed pixels of the map are shown, covering the same area the null
    test samples, so the two panels can be compared directly. Unobserved sky is
    left white (not grey) and a light grid is drawn over it.

    A compact single footprint is drawn zoomed in flat RA/Dec; a footprint that
    spans a large area or splits into fields far apart on the sky -- where a
    zoomed box would be mostly empty -- is drawn all-sky in Mollweide instead.
    Both rasterise the map to a fixed-size image the same way healpy's cartview
    does (one map lookup per output pixel), so the cost is set by the image size
    and does not grow with the map nside.

    Maps finer than ``plot_nside`` are degraded first, which only smooths the
    displayed values; it is not needed for speed. ``plot_nside=0`` keeps the
    native resolution.
    """
    import healpy as hp
    from matplotlib import colormaps

    # Degrade purely to smooth the displayed values; the projections below are
    # already nside-independent in cost.
    if plot_nside and nside > plot_nside:
        hsp_map = hsp_map.degrade(plot_nside, reduction="mean")
        nside = plot_nside

    pixels = hsp_map.valid_pixels
    if pixels.size == 0:
        ax.set_title(f"{title}\n(no observed pixels)")
        ax.axis("off")
        return

    ra, dec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("white")

    ra_span = _footprint_ra_span(ra)
    dec_span = float(dec.max() - dec.min())
    if ra_span > _WIDE_FOOTPRINT_DEG or dec_span > _WIDE_FOOTPRINT_DEG:
        _draw_allsky_map(ax, fig, hsp_map, nside, pixels, title, cmap, width)
    else:
        _draw_zoomed_map(ax, fig, hsp_map, ra, dec, title, cmap, width)


def _draw_zoomed_map(ax, fig, hsp_map, ra, dec, title, cmap, width):
    """
    Flat RA/Dec image zoomed to a compact footprint.

    RA is centred first so a footprint crossing RA=0 stays contiguous rather
    than splitting to the two edges; the axis then labels the true RA, wrapped
    back into [0, 360).
    """
    from matplotlib.ticker import FuncFormatter

    ra_center = np.degrees(np.arctan2(
        np.sin(np.radians(ra)).mean(), np.cos(np.radians(ra)).mean())) % 360.0
    dra = (ra - ra_center + 180.0) % 360.0 - 180.0
    ra_lo, ra_hi = ra_center + dra.min(), ra_center + dra.max()
    dec_lo, dec_hi = float(dec.min()), float(dec.max())

    # Size the image to the footprint's true sky proportions: a degree of RA
    # subtends cos(dec) less on the sky than a degree of Dec.
    cosd = np.cos(np.radians(0.5 * (dec_lo + dec_hi)))
    span_ra = max((ra_hi - ra_lo) * cosd, 1e-6)
    height = int(np.clip(round(width * (dec_hi - dec_lo) / span_ra), 2, 4 * width))

    # RA runs high -> low across the columns so it increases to the left, as on
    # the sky. Each image pixel takes the value of the map pixel covering it
    # (lookups wrapped into [0, 360)); positions off the footprint come back
    # invalid and are set to NaN so the colormap paints them white.
    ra_grid = np.linspace(ra_hi, ra_lo, width)
    dec_grid = np.linspace(dec_lo, dec_hi, height)
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    lon = RA.ravel() % 360.0
    vals = hsp_map.get_values_pos(lon, DEC.ravel(), lonlat=True)
    good = hsp_map.get_values_pos(lon, DEC.ravel(), lonlat=True, valid_mask=True)
    img = np.where(good, vals, np.nan).reshape(RA.shape).astype(float)

    im = ax.imshow(
        img,
        origin="lower",
        extent=(ra_hi, ra_lo, dec_lo, dec_hi),
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x % 360.0:g}"))
    ax.set_title(title)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.grid(True, color="0.7", lw=0.5, alpha=0.6)
    ax.set_axisbelow(False)  # imshow covers the footprint; keep the grid on top
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_allsky_map(ax, fig, hsp_map, nside, pixels, title, cmap, width):
    """
    All-sky Mollweide image for a wide or multi-field footprint.

    healpy's projector needs a full-sky array, whose size grows with nside, and
    an all-sky panel cannot resolve fine detail anyway, so the map is capped at
    a modest nside first. Meridians and parallels are projected through the same
    projector to draw the graticule, and the map's own axes are turned off.
    """
    import healpy as hp

    allsky_nside = min(nside, 256)
    if nside > allsky_nside:
        hsp_map = hsp_map.degrade(allsky_nside, reduction="mean")
        nside = allsky_nside
        pixels = hsp_map.valid_pixels

    m = np.full(hp.nside2npix(nside), hp.UNSEEN)
    m[pixels] = hsp_map[pixels].astype(float)
    proj = hp.projector.MollweideProj(xsize=2 * width)
    img = np.ma.masked_equal(
        proj.projmap(m, lambda x, y, z: hp.vec2pix(nside, x, y, z, nest=True)),
        hp.UNSEEN,
    )
    im = ax.imshow(img, extent=proj.get_extent(), origin="lower", cmap=cmap,
                   aspect="equal", interpolation="nearest")
    ax.axis("off")

    for lon in range(0, 360, 60):
        lat = np.linspace(-89.9, 89.9, 200)
        x, y = proj.ang2xy(np.full_like(lat, lon), lat, lonlat=True)
        ax.plot(x, y, color="0.7", lw=0.5, alpha=0.7, zorder=1)
    for lat in range(-60, 61, 30):
        lon = np.linspace(0.0, 360.0, 400)
        x, y = proj.ang2xy(lon, np.full_like(lon, lat), lonlat=True)
        ax.plot(x, y, color="0.7", lw=0.5, alpha=0.7, zorder=1)
    # Outer ellipse boundary of the Mollweide projection.
    t = np.linspace(0.0, 2 * np.pi, 400)
    ax.plot(2.0 * np.cos(t), np.sin(t), color="0.5", lw=0.8, zorder=1)

    ax.set_title(f"{title}  (all-sky)")
    # shrink so the bar tracks the short Mollweide ellipse, not the tall axes.
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.6)


class TXMeanShearSurveyProperties(PipelineStage):
    """
    Compute mean calibrated shear in bins of survey property maps.

    This null test checks whether the mean shear correlates with
    spatially-varying survey conditions such as PSF size, depth, or sky
    background. A non-zero trend indicates a potential systematic bias.

    Survey properties are read from aux_source_maps and aux_lens_maps.
    An optional directory of external .hs healsparse files can also be
    provided via the external_maps_dir config option.

    The aux_source_maps and aux_lens_maps inputs are optional: alias either (or
    both) to "none" in the pipeline to skip it. This lets the test run on the
    external_maps_dir maps alone, without having built the aux maps earlier.

    For each property the bin edges are chosen from the values it takes at the
    galaxies that enter the measurement -- galaxies in a source bin, on pixels
    the map observes -- so the bins span exactly the region being tested. Every
    property is therefore self-consistently binned and tested over its own
    observed sample, with no survey mask required.
    """

    name = "TXMeanShearSurveyProperties"
    parallel = True

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("aux_source_maps", MapsFile),
        ("aux_lens_maps", MapsFile),
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
        "properties": StageParameter(
            list, [],
            msg="Map names to test (e.g. ['psf/g1_2D', 'depth/depth']). Empty list means all available."
        ),
        "external_maps_dir": StageParameter(
            str, "",
            msg="Optional directory containing .hs healsparse files to also test"
        ),
        "map_plot_nside": StageParameter(
            int, 256,
            msg="Nside to degrade survey property maps to for the map panel "
                "(0 to plot at the map's native resolution)"
        ),
    }

    def run(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cat_type = read_shear_catalog_type(self)
        nbins = self.config["nbins"]
        chunk_rows = self.config["chunk_rows"]
        delta_gamma = self.config["delta_gamma"]
        outlier_fraction = self.config["outlier_fraction"]
        properties_filter = self.config["properties"]

        # Build shear column list for each catalog type.
        # For metadetect, ra/dec live in the unsheared variant subgroup, whose
        # name comes from META_VARIANTS[0] rather than being hard-coded.
        if cat_type == "metacal":
            shear_cols = [
                "ra", "dec",
                "g1", "g1_1p", "g1_1m", "g1_2p", "g1_2m",
                "g2", "g2_1p", "g2_2p", "g2_1m", "g2_2m",
                "weight",
            ]
            # No stored response is read: MetacalCalculator recomputes R per
            # survey-property bin from the sheared columns above, since the
            # response of a re-binned selection differs from the tomographic-bin
            # response saved in the catalog.
            ra_key, dec_key = "ra", "dec"
        elif cat_type == "metadetect":
            unsheared = META_VARIANTS[0]
            # ra/dec are read for every variant, not just the unsheared one:
            # each metadetect variant is a separate detection catalogue with its
            # own positions and length, and the property must be looked up at
            # each variant's positions (see _assign_property_column).
            shear_cols = metadetect_variants("ra", "dec", "g1", "g2", "weight")
            ra_key, dec_key = f"{unsheared}/ra", f"{unsheared}/dec"
        else:
            # lensfit, hsc
            shear_cols = ["ra", "dec", "g1", "g2", "weight", "m"]
            ra_key, dec_key = "ra", "dec"

        # MeanShearInBins cuts down to the source sample using a bin column.
        # For metadetect that column is per-variant (bin_ns, bin_1p, ...), since
        # the selector runs once per sheared variant; the other catalog types
        # have a single "bin". Compare TXDiagnosticPlots, which does the same.
        if cat_type == "metadetect":
            tomo_cols = [f"bin_{v}" for v in META_VARIANTS]
        else:
            tomo_cols = ["bin"]

        # Load all maps from the two aux map files, then optional external files.
        all_maps = {}  # name -> (hsp_map, nside)

        def _register_map(name, hsp_map, nside):
            if properties_filter and name not in properties_filter:
                return
            all_maps[name] = (hsp_map, nside)

        # Both aux map inputs are optional: aliased to "none" they are skipped,
        # letting the test run on external_maps_dir alone.
        for tag in ("aux_source_maps", "aux_lens_maps"):
            if self.get_input(tag) == "none":
                continue
            with self.open_input(tag, wrapper=True) as f:
                for name in f.list_maps():
                    hsp_map = f.read_map(name)
                    nside = f.read_map_info(name)["nside"]
                    _register_map(name, hsp_map, nside)

        if self.config["external_maps_dir"]:
            import healsparse
            root = self.config["external_maps_dir"]
            for path in sorted(glob.glob(f"{root}/*.hs")):
                name = pathlib.Path(path).stem
                hsp_map = healsparse.HealSparseMap.read(path)
                nside = hsp_map.nside_sparse
                _register_map(name, hsp_map, nside)

        if not all_maps:
            if self.rank == 0:
                print("TXMeanShearSurveyProperties: no survey property maps found.")
            return

        def _catalog_iterator():
            return self.combined_iterators(
                chunk_rows,
                "shear_catalog", "shear", shear_cols,
                "shear_tomography_catalog", "tomography", tomo_cols,
                longest=True,
            )

        # Galaxies in a source bin are the ones the measurement uses (the
        # MeanShearInBins source-bin cut). The bin column is per-variant for
        # metadetect -- the unsheared variant "ns" -- and a single "bin"
        # otherwise; see MeanShearInBins.selector.
        source_bin_key = "bin_ns" if cat_type == "metadetect" else "bin"

        # Pass 1: choose bin edges from the property values at the galaxy
        # positions that actually enter the measurement, i.e. galaxies in a
        # source bin where the map is observed. This is the same
        # {galaxies with defined shear in a source bin} ∩ {observed pixels}
        # region the mean shear is measured over, so the bins match the tested
        # region without needing a separate mask.
        sampled = {name: [] for name in all_maps}
        for s, e, data in _catalog_iterator():
            if self.rank == 0:
                print(f"TXMeanShearSurveyProperties: bin-edge pass rows {s:,} – {e:,}")
            in_source_bin = data[source_bin_key] != -1
            for name, (hsp_map, nside) in all_maps.items():
                vals = _property_values_at(
                    hsp_map, nside, data[ra_key], data[dec_key]
                )
                # Keep only source-sample galaxies with an observed value: this
                # is the sample the mean shear is binned over.
                keep = in_source_bin & np.isfinite(vals)
                sampled[name].append(vals[keep])

        # Combine the sampled values across chunks and MPI ranks, then set the
        # edges. allgather so every rank builds identical binners below (their
        # construction is collective through the calibrators).
        bin_edges = {}
        for name in list(all_maps):
            local = (
                np.concatenate(sampled[name]) if sampled[name] else np.empty(0)
            )
            del sampled[name]
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

        if not all_maps:
            if self.rank == 0:
                print(
                    "TXMeanShearSurveyProperties: no survey property map has a "
                    "usable range among the measured galaxies."
                )
            return

        # One MeanShearInBins accumulator per surviving map.
        binned_shears = {
            name: MeanShearInBins(
                "survey_prop",
                bin_edges[name],
                delta_gamma,
                cut_source_bin=True,
                shear_catalog_type=cat_type,
            )
            for name in all_maps
        }

        # Pass 2: accumulate mean shear in those bins, processing all maps per
        # chunk in a single sweep of the catalog.
        for s, e, data in _catalog_iterator():
            if self.rank == 0:
                print(f"TXMeanShearSurveyProperties: measurement pass rows {s:,} – {e:,}")
            for name, (hsp_map, nside) in all_maps.items():
                # Look up the property at each galaxy, with NaN for galaxies on
                # pixels the map does not observe (dropped by the binner). For
                # metadetect this writes one column per shear variant; see
                # _assign_property_column for why.
                _assign_property_column(
                    data, cat_type, hsp_map, nside, ra_key, dec_key
                )
                binned_shears[name].add_data(data)

        # Only rank 0 writes, so only rank 0 opens the outputs -- every rank
        # opening the same file would race. The loop below still runs on all
        # ranks, because binner.collect is collective and would deadlock if
        # some ranks skipped it.
        if self.rank == 0:
            output_dir = self.open_output("shear_survey_property_plots", wrapper=True)
            hdf_out = self.open_output("shear_survey_property_null")

        png_files = []
        for name, binner in binned_shears.items():
            mu, g1, g2, sigma1, sigma2 = binner.collect(self.comm)

            if self.rank != 0:
                continue

            safe_name = name.replace("/", "_")

            # Numerical results
            grp = hdf_out.create_group(safe_name)
            grp.create_dataset("bin_centers", data=mu)
            grp.create_dataset("g1", data=g1)
            grp.create_dataset("g2", data=g2)
            grp.create_dataset("sigma_g1", data=sigma1)
            grp.create_dataset("sigma_g2", data=sigma2)

            # Bins with a usable value in everything we plot. where_all_finite
            # returns indices, not a mask, and is the helper the other TXPipe
            # mean-shear null tests use.
            idx = where_all_finite(mu, g1, g2, sigma1, sigma2)

            # Two complementary statistics, as discussed for the other null
            # tests: chi2 against zero catches any departure from zero, while
            # the slope of a straight-line fit is the trend statistic the rest
            # of TXPipe quotes. Bins holding a single galaxy have sigma == 0
            # and are dropped from both, as they would divide by zero.
            chi2, n_dof = 0.0, 0
            fits = []
            for g_arr, s_arr in [(g1[idx], sigma1[idx]), (g2[idx], sigma2[idx])]:
                good = s_arr > 0
                chi2 += calc_chi2(g_arr[good], s_arr[good], np.zeros(int(good.sum())))
                n_dof += int(good.sum())
                slope, intercept, cov = fit_straight_line(
                    mu[idx][good], g_arr[good], y_err=s_arr[good]
                )
                fits.append((slope, intercept, cov[0, 0] ** 0.5))
            chi2_dof = chi2 / n_dof if n_dof > 0 else np.nan
            (slope1, intercept1, slope1_err), (slope2, intercept2, slope2_err) = fits

            grp.attrs["chi2"] = chi2
            grp.attrs["n_dof"] = n_dof
            grp.attrs["chi2_per_dof"] = chi2_dof
            grp.attrs["slope_g1"] = slope1
            grp.attrs["slope_g1_err"] = slope1_err
            grp.attrs["slope_g2"] = slope2
            grp.attrs["slope_g2_err"] = slope2_err

            # An empty plot is otherwise the only sign that a map produced no
            # usable bins at all, so say so explicitly.
            if idx.size == 0:
                print(
                    f"TXMeanShearSurveyProperties: WARNING - every bin is empty "
                    f"for '{name}'; the plot will have no points."
                )

            # Two panels: the property map itself on the left, so the trend on
            # the right can be read against the spatial structure driving it.
            fig = plt.figure(figsize=(12.0, 4.5))
            # The map panel is a touch narrower: a sky patch keeps its aspect
            # ratio, so given equal widths it would sit small beside the trend.
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])

            map_ax = fig.add_subplot(gs[0])
            hsp_map, map_nside = all_maps[name]
            _plot_property_map(
                map_ax, fig, hsp_map, map_nside, safe_name,
                plot_nside=self.config["map_plot_nside"],
            )

            ax = fig.add_subplot(gs[1])
            # Small horizontal offset so the g1/g2 series don't overlap, taken
            # over the finite bins only: empty bins (NaN from collect) would
            # otherwise poison it and blank the whole scatter. g1 sits to the
            # right of each bin and g2 to the left, matching TXDiagnosticPlots.
            dx = _marker_offset(mu[idx])
            ax.axhline(0, color="k", lw=0.8, ls="--")
            # Error bars are drawn in black *on top of* the markers rather than
            # in the series colour underneath them. A bin holding a single
            # galaxy has sigma exactly zero, and one holding two or three that
            # happen to agree has a sigma far smaller than the marker: both look
            # identical when the bar is hidden behind the point. Drawn this way,
            # a true zero shows no black at all, while a tiny error shows a
            # short black line through the marker.
            for x_off, g_arr, s_arr in [
                (mu[idx] + dx, g1[idx], sigma1[idx]),
                (mu[idx] - dx, g2[idx], sigma2[idx]),
            ]:
                ax.errorbar(
                    x_off, g_arr, s_arr,
                    fmt="none", ecolor="k", elinewidth=0.8, zorder=3,
                )
            ax.plot(
                mu[idx] + dx, g1[idx], "s", markersize=4, color="tab:blue",
                linestyle="none", zorder=2,
                label=f"g1  (m={slope1:.2e} $\\pm$ {slope1_err:.2e})",
            )
            ax.plot(
                mu[idx] - dx, g2[idx], "o", markersize=4, color="tab:orange",
                linestyle="none", zorder=2,
                label=f"g2  (m={slope2:.2e} $\\pm$ {slope2_err:.2e})",
            )

            # The fitted lines, drawn across the range that was actually fitted.
            if idx.size > 1:
                x_line = np.array([mu[idx].min(), mu[idx].max()])
                ax.plot(x_line, slope1 * x_line + intercept1, color="tab:blue", lw=1)
                ax.plot(x_line, slope2 * x_line + intercept2, color="tab:orange", lw=1)

            ax.set_xlabel(safe_name)
            ax.set_ylabel("Mean shear")
            ax.set_title(f"{safe_name}  (χ²/dof = {chi2_dof:.2f})")
            ax.legend(fontsize=8)
            # Explicit spacing rather than tight_layout: the map colorbar and
            # the trend panel need a wide gap between them to stop the map's
            # Dec axis crowding the trend's y-axis.
            fig.subplots_adjust(left=0.06, right=0.96, bottom=0.15, wspace=0.32)

            png_name = f"{safe_name}.png"
            fig.savefig(output_dir.path_for_file(png_name))
            plt.close(fig)
            png_files.append(png_name)

        if self.rank == 0:
            output_dir.write_listing(png_files)
            hdf_out.close()
