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


def _in_footprint(ra, dec, mask):
    """
    Boolean array flagging which (ra, dec) positions fall inside the mask
    footprint.

    ``mask`` is a boolean HealSparse map (as returned by
    ``read_mask(..., returnbool=True)``). Positions outside the footprint land
    on unobserved pixels, for which HealSparse returns the ``False`` sentinel,
    so a direct lookup gives the in/out flag we want.
    """
    import healpy as hp

    pix = hp.ang2pix(mask.nside_sparse, ra, dec, lonlat=True, nest=True)
    return mask[pix]


def _compute_bin_edges(hsp_map, nside, nbins, outlier_fraction, mask):
    """
    Compute bin edges for a survey property map, restricted to the mask
    footprint.

    Only map pixels that lie inside ``mask`` contribute to the percentile
    range, so per-band coverage that extends beyond the shear footprint does
    not skew the binning.

    Maps taking only a few distinct values (integer counts such as
    ``bright_objects/count``, or flag maps) are binned with one bin per value,
    centred on that value. Evenly spaced edges would instead land exactly *on*
    the integers, and since ``MeanShearInBins.selector`` selects with strict
    inequalities on both sides, every galaxy would then fall into no bin at all
    and the test would silently come out empty.

    Returns ``None`` if the map has no usable range inside the footprint --
    because it does not overlap the footprint, or because it is constant there
    and so carries no trend to test. That map is then skipped.
    """
    import healpy as hp

    valid = hsp_map.valid_pixels
    # Restrict to pixels inside the footprint before choosing the range.
    ra, dec = hp.pix2ang(nside, valid, nest=True, lonlat=True)
    valid = valid[_in_footprint(ra, dec, mask)]
    if valid.size == 0:
        return None

    vals = hsp_map[valid].astype(float)
    vals = vals[np.isfinite(vals)]
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
        # Constant inside the footprint: no trend can be measured.
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


def _property_values_at(hsp_map, nside, ra, dec, in_footprint):
    """
    Look up a survey property map at galaxy positions, returning float values
    with ``NaN`` for galaxies that should be excluded from the test.

    A value is set to ``NaN`` when the galaxy falls either

    * on a pixel the map does not observe, or
    * outside the survey footprint (``in_footprint`` is False).

    Unobserved pixels are detected with HealSparse's own validity mask rather
    than by comparing against ``hp.UNSEEN``: the unobserved sentinel differs by
    dtype (``hp.UNSEEN`` for float64, but a distinct value for integer and
    float32 maps), so an equality test would silently miss them. ``NaN`` values
    fall outside every bin and are dropped by ``MeanShearInBins``.
    """
    import healpy as hp

    pixels = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    # valid_mask=True flags observed pixels for any map dtype.
    valid = hsp_map.get_values_pix(pixels, valid_mask=True)
    prop_vals = hsp_map[pixels].astype(float)
    prop_vals[~valid] = np.nan
    prop_vals[~in_footprint] = np.nan
    return prop_vals


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


def _plot_property_map(hsp_map, nside, mask, title, plot_nside=0):
    """
    Draw a survey property map into the current matplotlib axes.

    Only pixels inside ``mask`` are shown, so the map covers the same footprint
    the null test is computed over and the two panels can be compared directly.

    Maps finer than ``plot_nside`` are degraded first: healpy needs a full-sky
    array to plot, which is large at high nside, and the extra detail is not
    visible at plot size anyway. ``plot_nside=0`` keeps the native resolution.
    """
    import healpy as hp
    import matplotlib.pyplot as plt

    # Degrade before building the full-sky array, not after.
    if plot_nside and nside > plot_nside:
        hsp_map = hsp_map.degrade(plot_nside, reduction="mean")
        nside = plot_nside

    pixels = hsp_map.valid_pixels
    ra, dec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
    keep = _in_footprint(ra, dec, mask)
    pixels, ra, dec = pixels[keep], ra[keep], dec[keep]

    if pixels.size == 0:
        plt.title(f"{title}\n(no pixels inside footprint)")
        plt.axis("off")
        return

    m = np.full(hp.nside2npix(nside), hp.UNSEEN)
    m[pixels] = hsp_map[pixels].astype(float)

    # Zoom to the footprint rather than showing the whole sky.
    lonra = np.clip([ra.min() - 0.1, ra.max() + 0.1], 0.0, 360.0)
    latra = np.clip([dec.min() - 0.1, dec.max() + 0.1], -90.0, 90.0)
    hp.cartview(
        m,
        lonra=list(lonra),
        latra=list(latra),
        title=title,
        hold=True,
        nest=True,
        # cartview pads generously by default, which leaves the map panel much
        # smaller than the plot panel beside it.
        margins=(0.02, 0.02, 0.02, 0.04),
    )


class TXMeanShearSurveyProperties(PipelineStage):
    """
    Compute mean calibrated shear in bins of survey property maps.

    This null test checks whether the mean shear correlates with
    spatially-varying survey conditions such as PSF size, depth, or sky
    background. A non-zero trend indicates a potential systematic bias.

    Survey properties are read from aux_source_maps and aux_lens_maps.
    An optional directory of external .hs healsparse files can also be
    provided via the external_maps_dir config option.

    The test is restricted to the survey footprint defined by the mask input:
    galaxies outside the footprint, and map pixels outside it when choosing bin
    edges, are excluded. This ensures every survey property is tested over the
    same region even when the individual property maps (e.g. in different bands)
    have different coverage.
    """

    name = "TXMeanShearSurveyProperties"
    parallel = True

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("aux_source_maps", MapsFile),
        ("aux_lens_maps", MapsFile),
        ("mask", MapsFile),
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
            msg="Fraction of map pixels excluded as outliers at each tail when computing bin edges"
        ),
        "mask_threshold": StageParameter(
            float, 0.0,
            msg="Minimum fractional coverage for a mask pixel to count as inside the footprint"
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
            extra_iters = ["shear_tomography_catalog", "response", ["R_gamma"]]
            ra_key, dec_key = "ra", "dec"
        elif cat_type == "metadetect":
            unsheared = META_VARIANTS[0]
            shear_cols = [f"{unsheared}/ra", f"{unsheared}/dec"] + metadetect_variants(
                "g1", "g2", "weight"
            )
            extra_iters = []
            ra_key, dec_key = f"{unsheared}/ra", f"{unsheared}/dec"
        else:
            # lensfit, hsc
            shear_cols = ["ra", "dec", "g1", "g2", "weight", "m"]
            extra_iters = []
            ra_key, dec_key = "ra", "dec"

        # MeanShearInBins cuts down to the source sample using a bin column.
        # For metadetect that column is per-variant (bin_ns, bin_1p, ...), since
        # the selector runs once per sheared variant; the other catalog types
        # have a single "bin". Compare TXDiagnosticPlots, which does the same.
        if cat_type == "metadetect":
            tomo_cols = [f"bin_{v}" for v in META_VARIANTS]
        else:
            tomo_cols = ["bin"]

        # Read the survey mask as a boolean footprint map. The null test is only
        # computed for galaxies (and map pixels) inside this footprint, so that
        # every survey property is tested over the same region even when the
        # individual property maps have different (e.g. per-band) coverage.
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_mask(
                thresh=self.config["mask_threshold"], returnbool=True
            )

        # Load all maps from the two aux map files, then optional external files.
        all_maps = {}  # name -> (hsp_map, nside, bin_edges)

        def _register_map(name, hsp_map, nside):
            if properties_filter and name not in properties_filter:
                return
            edges = _compute_bin_edges(hsp_map, nside, nbins, outlier_fraction, mask)
            if edges is None:
                if self.rank == 0:
                    print(
                        f"TXMeanShearSurveyProperties: skipping '{name}' "
                        "(no usable range inside the mask footprint)."
                    )
                return
            all_maps[name] = (hsp_map, nside, edges)

        for tag in ("aux_source_maps", "aux_lens_maps"):
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

        # One MeanShearInBins accumulator per map.
        binned_shears = {
            name: MeanShearInBins(
                "survey_prop",
                edges,
                delta_gamma,
                cut_source_bin=True,
                shear_catalog_type=cat_type,
            )
            for name, (_, _, edges) in all_maps.items()
        }

        # Single pass through the shear catalog, processing all maps per chunk.
        it = self.combined_iterators(
            chunk_rows,
            "shear_catalog", "shear", shear_cols,
            "shear_tomography_catalog", "tomography", tomo_cols,
            *extra_iters,
            longest=True,
        )

        for s, e, data in it:
            if self.rank == 0:
                print(f"TXMeanShearSurveyProperties: rows {s:,} – {e:,}")
            # Galaxies outside the mask footprint are excluded from every map's
            # test, giving all survey properties a single consistent footprint.
            in_footprint = _in_footprint(data[ra_key], data[dec_key], mask)
            for name, (hsp_map, nside, _) in all_maps.items():
                # Look up the property at each galaxy, with NaN for galaxies on
                # unobserved pixels or outside the footprint (dropped by the
                # binner). See _property_values_at for the dtype handling.
                data["survey_prop"] = _property_values_at(
                    hsp_map, nside, data[ra_key], data[dec_key], in_footprint
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
            fig = plt.figure(figsize=(11.5, 4.5))
            # The map panel is narrower: a sky patch keeps its aspect ratio, so
            # given equal widths it would sit small in a sea of white space.
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45])

            plt.subplot(gs[0])
            hsp_map, map_nside, _ = all_maps[name]
            _plot_property_map(
                hsp_map, map_nside, mask, safe_name,
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
            # Not tight_layout: healpy's cartview axes are not compatible with
            # it and it warns and mislays the panels.
            fig.subplots_adjust(left=0.02, right=0.97, bottom=0.15, wspace=0.12)

            png_name = f"{safe_name}.png"
            fig.savefig(output_dir.path_for_file(png_name))
            plt.close(fig)
            png_files.append(png_name)

        if self.rank == 0:
            output_dir.write_listing(png_files)
            hdf_out.close()
