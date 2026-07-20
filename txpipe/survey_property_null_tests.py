import glob
import pathlib
import numpy as np
from ceci.config import StageParameter

from .base_stage import PipelineStage
from .data_types import ShearCatalog, TomographyCatalog, MapsFile, FileCollection, HDFFile
from .shear_calibration import MeanShearInBins, metadetect_variants
from .utils import read_shear_catalog_type


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
    not skew the binning. Returns ``None`` if the map does not overlap the
    footprint (that map is then skipped).
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
        # For metadetect, ra/dec live in the "00" (unsheared) variant subgroup.
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
            shear_cols = ["00/ra", "00/dec"] + metadetect_variants("g1", "g2", "weight")
            extra_iters = []
            ra_key, dec_key = "00/ra", "00/dec"
        else:
            # lensfit, hsc
            shear_cols = ["ra", "dec", "g1", "g2", "weight", "m"]
            extra_iters = []
            ra_key, dec_key = "ra", "dec"

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
                        "(no overlap with mask footprint)."
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
            "shear_tomography_catalog", "tomography", ["bin"],
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

        # Collect across MPI ranks and write outputs (rank 0 writes).
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

            # Plot with chi2/dof annotation
            idx = np.isfinite(mu) & np.isfinite(g1) & np.isfinite(g2)
            chi2, n_dof = 0.0, 0
            for g_arr, s_arr in [(g1[idx], sigma1[idx]), (g2[idx], sigma2[idx])]:
                good = s_arr > 0
                chi2 += np.sum((g_arr[good] / s_arr[good]) ** 2)
                n_dof += int(good.sum())
            chi2_dof = chi2 / n_dof if n_dof > 0 else np.nan

            fig, ax = plt.subplots(figsize=(7, 4))
            # Small horizontal offset so the g1/g2 series don't overlap. Derive
            # it from the finite bins only: empty bins (NaN from collect) at the
            # start would otherwise poison dx and blank the whole scatter.
            dx = 0.1 * np.median(np.diff(np.sort(mu[idx]))) if idx.sum() > 1 else 0
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.errorbar(
                mu[idx] + dx, g1[idx], sigma1[idx],
                fmt="s", markersize=4, label="g1", color="tab:blue",
            )
            ax.errorbar(
                mu[idx] - dx, g2[idx], sigma2[idx],
                fmt="o", markersize=4, label="g2", color="tab:orange",
            )
            ax.set_xlabel(safe_name)
            ax.set_ylabel("Mean shear")
            ax.set_title(f"{safe_name}  (χ²/dof = {chi2_dof:.2f})")
            ax.legend(fontsize=8)
            fig.tight_layout()

            png_name = f"{safe_name}.png"
            fig.savefig(output_dir.path_for_file(png_name))
            plt.close(fig)
            png_files.append(png_name)

        if self.rank == 0:
            output_dir.write_listing(png_files)

        hdf_out.close()
