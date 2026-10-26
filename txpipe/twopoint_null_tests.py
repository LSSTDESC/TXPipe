from .base_stage import PipelineStage
from .data_types import (
    HDFFile,
    MapsFile,
    TomographyCatalog,
    RandomsCatalog,
    SACCFile,
    PNGFile,
    TextFile,
    QPNOfZFile,
    BinnedCatalog,
    FiducialCosmology,
)
import numpy as np
from .twopoint import TXTwoPoint, SHEAR_SHEAR, SHEAR_POS, POS_POS, TREECORR_CONFIG
from .utils import DynamicSplitter
from .random_cats import TXRandomCat
from ceci.config import StageParameter


class TXStarCatalogSplitter(PipelineStage):
    """
    Split a star catalog into bright and dim stars
    """

    name = "TXStarCatalogSplitter"
    parallel = False

    inputs = [
        ("star_catalog", HDFFile),
    ]

    outputs = [
        ("binned_star_catalog", HDFFile),
    ]

    config_options = {
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
        "initial_size": StageParameter(int, 100_000, msg="Initial size for star catalog bins."),
        "mag_col": StageParameter(str, "mag_r", msg="Magnitude column used to split bright from dim stars."),
        "bright_mag_min": StageParameter(float, 14.0, msg="Bright star bin lower magnitude limit."),
        "bright_mag_max": StageParameter(float, 18.2, msg="Bright star bin upper magnitude limit."),
        "dim_mag_min": StageParameter(float, 18.2, msg="Dim star bin lower magnitude limit."),
        "dim_mag_max": StageParameter(float, 22.0, msg="Dim star bin upper magnitude limit."),
    }

    def run(self):
        cols = ["ra", "dec"]
        mag_col = self.config["mag_col"]

        #  Object we use to make the separate lens bins catalog
        output = self.open_output("binned_star_catalog")
        group = output.create_group("stars")
        group.attrs["nbin"] = 2
        group.attrs["nbin_lens"] = 2

        bins = {
            "bright": self.config["initial_size"],
            "dim": self.config["initial_size"],
        }

        splitter = DynamicSplitter(group, "bin", cols, bins)

        # Also read the magnitude.  We don't save it as we don't
        # need it later (?) but we do use it for the split
        it = self.iterate_hdf(
            "star_catalog",
            "stars",
            cols + [mag_col],
            self.config["chunk_rows"],
        )

        for s, e, data in it:
            print(f"Process 0 binning data in range {s:,} - {e:,}")
            mag = data.pop(mag_col)
            dim_cut = (mag > self.config["dim_mag_min"]) & (mag < self.config["dim_mag_max"])
            bright_cut = (mag > self.config["bright_mag_min"]) & (mag < self.config["bright_mag_max"])

            dim = {name: col[dim_cut] for name, col in data.items()}
            bright = {name: col[bright_cut] for name, col in data.items()}

            splitter.write_bin(dim, "dim")
            splitter.write_bin(bright, "bright")

        splitter.finish()
        output.close()


class TXStarRandomCat(TXRandomCat):
    """
    Generate randoms covering the star catalog footprint, for TXGammaTStars

    A wrapper around TXRandomCat for source-only pipelines, which have no
    depth map or lens n(z). It writes placeholder versions of those inputs
    and then runs TXRandomCat unchanged:

    - a constant depth map, so the density is the configured ``density`` everywhere
    - a flat n(z), since gamma_t around stars does not use the random redshifts
    - the mask restricted to the coarse pixels that contain at least one star,
      so that fields with no stars get no randoms

    The stars should be cut to the same mask, e.g. with TXCutCatalog.
    """

    name = "TXStarRandomCat"
    # The placeholder files are written to a local temporary directory
    parallel = False

    inputs = [
        ("mask", MapsFile),
        ("binned_star_catalog", BinnedCatalog),
        ("fiducial_cosmology", FiducialCosmology),
    ]

    config_options = TXRandomCat.config_options | {
        "coverage_nside": StageParameter(
            int, 256, msg="Nside of the star occupancy map used to decide which parts of the mask the stars cover."
        ),
    }

    # TXRandomCat inputs that we replace with placeholder files
    placeholder_types = {"depth_map": MapsFile, "lens_photoz_stack": QPNOfZFile, "mask": MapsFile}
    placeholder_paths = None

    def run(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.placeholder_paths = self.write_placeholder_inputs(tmp)
            super().run()

    def get_input(self, tag):
        if self.placeholder_paths and tag in self.placeholder_paths:
            return self.placeholder_paths[tag]
        return super().get_input(tag)

    def get_input_type(self, tag):
        if self.placeholder_paths and tag in self.placeholder_paths:
            return self.placeholder_types[tag]
        return super().get_input_type(tag)

    def write_placeholder_inputs(self, tmp):
        import healpy
        import healsparse
        import qp

        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")
            metadata = dict(f.file["maps/mask"].attrs)
        nside = mask.nside_sparse
        coverage_nside = self.config["coverage_nside"]
        if coverage_nside > nside:
            raise ValueError(f"coverage_nside ({coverage_nside}) must not be larger than the mask nside ({nside})")

        # Coarse pixels that contain at least one star in either bin
        star_pix = []
        with self.open_input("binned_star_catalog") as f:
            for b in ["bright", "dim"]:
                g = f[f"stars/bin_{b}"]
                star_pix.append(healpy.ang2pix(coverage_nside, g["ra"][:], g["dec"][:], lonlat=True, nest=True))
        star_pix = np.unique(np.concatenate(star_pix))

        # Keep the mask pixels whose parent coarse pixel has stars. In the
        # NEST scheme each halving of nside drops two bits of pixel index.
        pix = mask.valid_pixels
        shift = 2 * int(np.log2(nside // coverage_nside))
        pix = pix[np.isin(pix >> shift, star_pix)]
        print(f"Stars cover {pix.size:,} of the {mask.n_valid:,} mask pixels")

        star_mask = healsparse.HealSparseMap.make_empty_like(mask)
        star_mask[pix] = mask[pix]

        # Any constant depth gives the configured density everywhere
        depth = healsparse.HealSparseMap.make_empty(mask.nside_coverage, nside, np.float64)
        depth[pix] = 25.0
        depth_band = self.config["depth_band"]
        depth_name = f"depth/depth_{depth_band}" if depth_band else "depth/depth"

        # One flat tomographic bin plus the 2D n(z)
        n_of_z = qp.Ensemble(qp.interp, data=dict(xvals=np.linspace(0.0, 3.0, 31), yvals=np.ones((2, 31))))

        paths = {tag: f"{tmp}/{tag}.hdf5" for tag in self.placeholder_types}
        with MapsFile(paths["mask"], "w") as f:
            f.write_map("mask", star_mask, metadata)
        with MapsFile(paths["depth_map"], "w") as f:
            f.write_map(depth_name, depth, metadata)
        with QPNOfZFile(paths["lens_photoz_stack"], "w") as f:
            f.write_ensemble(n_of_z)
        return paths


class TXGammaTFieldCenters(TXTwoPoint):
    """
    Make diagnostic 2pt measurements of tangential shear around field centers

    This subclass of the standard TXTwoPoint uses the centers
    of exposure fields as "lenses", as a systematics test.
    """

    name = "TXGammaTFieldCenters"
    inputs = [
        ("binned_shear_catalog", BinnedCatalog),
        ("random_cats", RandomsCatalog),
        ("exposures", HDFFile),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [
        ("gammat_field_center", SACCFile),
        ("gammat_field_center_plot", PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = TREECORR_CONFIG | {
        "calcs": StageParameter(
            list, [0, 1, 2], msg="Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position"
        ),
        "reduce_randoms_size": StageParameter(float, 1.0, msg="Factor to reduce the size of random catalogs."),
        "var_method": StageParameter(str, "shot", msg="Method for computing variance (shot, jackknife, etc.)."),
        "npatch": StageParameter(int, 5, msg="Number of patches for null tests."),
        "use_true_shear": StageParameter(bool, False, msg="Whether to use true shear values."),
        "subtract_mean_shear": StageParameter(bool, False, msg="Whether to subtract mean shear."),
        "use_randoms": StageParameter(bool, True, msg="Whether to use random catalogs."),
        "patch_dir": StageParameter(str, "./cache/patches", msg="Directory for storing patch files."),
        "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
        "share_patch_files": StageParameter(bool, False, msg="Whether to share patch files across processes."),
        "use_subsampled_randoms": StageParameter(bool, False, msg="Use subsampled randoms file for RR calculation."),
    }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib

        matplotlib.use("agg")
        super().run()

    def read_nbin(self):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        return ["all"], [0]

    def get_random_catalog(self, i):
        import treecorr

        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("random_cats"),
            ext="randoms",
            ra_col="ra",
            dec_col="dec",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("random_cats", i),
        )
        return rancat

    def get_lens_catalog(self, i):
        import treecorr

        # check if we are using the old format exposures file
        # or the new one
        with self.open_input("exposures", wrapper=False) as f:
            if "visits" in f.keys():
                # This is the new format
                ext = "visits"
                ra_col = "ra"
                dec_col = "dec"
            else:
                # This is the old format
                ext = "exposures"
                ra_col = "ratel"
                dec_col = "dectel"

        assert i == 0
        cat = treecorr.Catalog(
            self.get_input("exposures"),
            ext=ext,
            ra_col=ra_col,
            dec_col=dec_col,
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("exposures", i),
        )
        return cat

    def select_calculations(self, source_list, lens_list):
        # We only want a single calculation, the 2D gamma_T around
        # the field centers
        return [("all", 0, SHEAR_POS)]

    def write_output(self, source_list, lens_list, meta, results):
        # This subclass only needs the root process for this task
        if self.rank != 0:
            return
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt

        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output("gammat_field_center_plot", wrapper=True)

        plt.errorbar(dtheta, dtheta * dvalue, derror, fmt="ro", capsize=3)
        plt.xscale("log")

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Field Center Tangential Shear")

        fig.close()

    def write_output_sacc(self, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc

        dt = "galaxyFieldCenter_shearDensity_xi_t"

        S = sacc.Sacc()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer("Misc", "fieldcenter")
        S.add_tracer("Misc", "source2d")

        d = results[0]
        assert len(results) == 1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(
                dt,
                ("source2d", "fieldcenter"),
                dvalue[i],
                theta=dtheta[i],
                error=derror[i],
                npair=dnpair[i],
                weight=dweight[i],
            )

        self.write_metadata(S, meta)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output("gammat_field_center"), overwrite=True)


class TXGammaTStars(TXTwoPoint):
    """
    Make diagnostic 2pt measurements of tangential shear around stars

    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """

    name = "TXGammaTStars"
    inputs = [
        ("binned_shear_catalog", BinnedCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("random_cats", RandomsCatalog),
        ("binned_star_catalog", BinnedCatalog),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
        ("binned_random_catalog", HDFFile),
    ]
    outputs = [
        ("gammat_bright_stars", SACCFile),
        ("gammat_bright_stars_plot", PNGFile),
        ("gammat_dim_stars", SACCFile),
        ("gammat_dim_stars_plot", PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = TREECORR_CONFIG | {
        "calcs": StageParameter(
            list, [0, 1, 2], msg="Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position"
        ),
        "reduce_randoms_size": StageParameter(float, 1.0, msg="Factor to reduce the size of random catalogs."),
        "var_method": StageParameter(str, "shot", msg="Method for computing variance (shot, jackknife, etc.)."),
        "npatch": StageParameter(int, 5, msg="Number of patches for null tests."),
        "use_true_shear": StageParameter(bool, False, msg="Whether to use true shear values."),
        "subtract_mean_shear": StageParameter(bool, False, msg="Whether to subtract mean shear."),
        "use_randoms": StageParameter(bool, True, msg="Whether to use random catalogs."),
        "patch_dir": StageParameter(str, "./cache/patches", msg="Directory for storing patch files."),
        "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
        "share_patch_files": StageParameter(bool, False, msg="Whether to share patch files across processes."),
        "use_subsampled_randoms": StageParameter(bool, False, msg="Use subsampled randoms file for RR calculation."),
    }

    def run(self):
        import matplotlib

        matplotlib.use("agg")
        super().run()

    def read_nbin(self):
        # We use two sets of stars, dim and bright
        return ["all"], ["bright", "dim"]

    def get_lens_catalog(self, i):
        import treecorr

        assert i == "bright" or i == "dim"
        cat = treecorr.Catalog(
            self.get_input("binned_star_catalog"),
            ext=f"stars/bin_{i}",
            ra_col="ra",
            dec_col="dec",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_star_catalog", i),
        )
        return cat

    def get_random_catalog(self, i):
        import treecorr

        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("random_cats"),
            ext="randoms",
            ra_col="ra",
            dec_col="dec",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("random_cats", i),
        )
        return rancat

    def select_calculations(self, source_list, lens_list):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [("all", "bright", SHEAR_POS), ("all", "dim", SHEAR_POS)]

    def write_output(self, source_list, lens_list, meta, results):
        # This subclass only needs the root process for this task
        if self.rank != 0:
            return

        # we write output both to file for later and to a plot
        self.write_output_sacc(meta, results[0], "gammat_bright_stars", "Bright")
        self.write_output_sacc(meta, results[1], "gammat_dim_stars", "Dim")
        self.write_output_plot(results[0], "gammat_bright_stars_plot", "Bright")
        self.write_output_plot(results[1], "gammat_dim_stars_plot", "Dim")

    def write_output_plot(self, d, image_file, text):
        import matplotlib.pyplot as plt

        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dvalue_x = d.object.xi_im
        cov_x = self.gamma_x_cov(d)
        derror_x = np.sqrt(np.diag(cov_x))

        fig = self.open_output(image_file, wrapper=True)

        # compute the chi^2 against zero
        chi2, pte, ndof = self.null_chi2(d, dvalue, d.object.cov, resampled=True)
        chi2_x, pte_x, _ = self.null_chi2(d, dvalue_x, cov_x, resampled=True)
        print(f"{text} stars: gamma_t chi^2 = {chi2:.2f}, gamma_x chi^2 = {chi2_x:.2f} for {ndof} dof")

        plt.errorbar(
            dtheta,
            dtheta * dvalue,
            dtheta * derror,
            fmt="ro",
            capsize=3,
            label=rf"$\gamma_t$: $\chi^2/dof = {chi2:.1f}/{ndof}$, PTE = {pte:.2f}",
        )
        # Offset gamma_x slightly so the error bars do not overlap
        plt.errorbar(
            dtheta * 1.05,
            dtheta * dvalue_x,
            dtheta * derror_x,
            fmt="bs",
            capsize=3,
            label=rf"$\gamma_\times$ ($\gamma_t$ cov): $\chi^2/dof = {chi2_x:.1f}/{ndof}$, PTE = {pte_x:.2f}",
        )
        plt.axhline(0, color="k", lw=0.5)
        plt.legend(loc="best")
        plt.xscale("log")

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma(\theta)$")
        plt.title(f"{text} Star Centers Tangential Shear")
        plt.tight_layout()

        fig.close()

    def gamma_x_cov(self, d):
        # Use the gamma_t covariance for gamma_x too. With no signal around
        # stars both components see the same shape noise and edge effects.
        # treecorr 5.1 cannot resample gamma_x itself when randoms are
        # attached, as it does not subtract them from gamma_x in each sample,
        # and shot noise misses the large-scale scatter in small fields.
        return d.object.cov

    def null_chi2(self, d, value, cov, resampled):
        """
        chi^2 of value against zero, its PTE, and the number of bins used
        """
        import scipy.stats
        from .utils.fitting import calc_chi2

        # Empty bins have zero value and variance, so leave them out
        use = d.object.npairs > 0
        ndof = int(use.sum())
        chi2 = calc_chi2(value[use], cov[np.ix_(use, use)], np.zeros(ndof))

        # Debias the inverse of a resampled covariance (Hartlap et al 2007)
        method = self.config["var_method"]
        if resampled and method != "shot":
            nsample = d.object.num_bootstrap if "bootstrap" in method else d.object.npatch1
            chi2 *= (nsample - ndof - 2) / (nsample - 1) if nsample > ndof + 2 else np.nan

        return chi2, scipy.stats.chi2.sf(chi2, ndof), ndof

    def write_output_sacc(self, meta, d, sacc_file, text):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc

        dt = "galaxyStar_shearDensity_xi_t"

        S = sacc.Sacc()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        name = "{}_stars".format(text.lower())
        S.add_tracer("Misc", name)
        S.add_tracer("Misc", "source2d")

        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(
                dt,
                ("source2d", name),
                dvalue[i],
                theta=dtheta[i],
                error=derror[i],
                npair=dnpair[i],
                weight=dweight[i],
            )

        # The cross component, which should also be zero
        dvalue_x = d.object.xi_im
        cov_x = self.gamma_x_cov(d)
        derror_x = np.sqrt(np.diag(cov_x))
        for i in range(n):
            S.add_data_point(
                "galaxyStar_shearDensity_xi_x",
                ("source2d", name),
                dvalue_x[i],
                theta=dtheta[i],
                error=derror_x[i],
                npair=dnpair[i],
                weight=dweight[i],
            )

        # Block-diagonal, as in TXTwoPoint.add_gamma_x_data_points
        S.add_covariance([d.object.cov, cov_x])

        self.write_metadata(S, meta)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output(sacc_file), overwrite=True)


class TXGammaTRandoms(TXTwoPoint):
    """
    Make diagnostic 2pt measurements of tangential shear around randoms

    It's not clear to me that this is a useful null test; if it was we
    wouldn't need to subtrac this term in the Landay-Szalay estimator.

    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """

    name = "TXGammaTRandoms"
    inputs = [
        ("binned_shear_catalog", BinnedCatalog),
        ("random_cats", RandomsCatalog),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [
        ("gammat_randoms", SACCFile),
        ("gammat_randoms_plot", PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = TREECORR_CONFIG | {
        "calcs": StageParameter(
            list, [0, 1, 2], msg="Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position"
        ),
        "reduce_randoms_size": StageParameter(float, 1.0, msg="Factor to reduce the size of random catalogs."),
        "var_method": StageParameter(str, "shot", msg="Method for computing variance (shot, jackknife, etc.)."),
        "npatch": StageParameter(int, 5, msg="Number of patches for null tests."),
        "use_true_shear": StageParameter(bool, False, msg="Whether to use true shear values."),
        "subtract_mean_shear": StageParameter(bool, False, msg="Whether to subtract mean shear."),
        "use_randoms": StageParameter(bool, False, msg="Whether to use random catalogs."),
        "patch_dir": StageParameter(str, "./cache/patches", msg="Directory for storing patch files."),
        "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
        "share_patch_files": StageParameter(bool, False, msg="Whether to share patch files across processes."),
        "use_subsampled_randoms": StageParameter(bool, False, msg="Use subsampled randoms file for RR calculation."),
    }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib

        matplotlib.use("agg")
        super().run()

    def read_nbin(self):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        return ["all"], [0]

    def get_random_catalog(self, i):
        # override the parent method
        # so that we don't load the randoms here,
        # because if we subtract randoms from randoms
        # we get nothing.
        return None

    def get_lens_catalog(self, i):
        import treecorr

        assert i == 0
        cat = treecorr.Catalog(
            self.get_input("random_cats"),
            ext=f"randoms",
            ra_col="ra",
            dec_col="dec",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("random_cats", i),
        )
        return cat

    def select_calculations(self, source_list, lens_list):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [("all", 0, SHEAR_POS)]

    def write_output(self, source_list, lens_list, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt

        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output("gammat_randoms_plot", wrapper=True)

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (dvalue - flat1) / derror
        chi2 = np.sum(z**2)
        chi2dof = chi2 / (len(dtheta) - 1)

        plt.errorbar(
            dtheta,
            dtheta * dvalue,
            dtheta * derror,
            fmt="ro",
            capsize=3,
            label=r"$\chi^2/dof = $" + str(chi2dof),
        )
        plt.legend(loc="best")
        plt.xscale("log")

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Randoms Tangential Shear")

        fig.close()

    def write_output_sacc(self, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc

        dt = "galaxyRandoms_shearDensity_xi_t"

        S = sacc.Sacc()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer("Misc", "randoms")
        S.add_tracer("Misc", "source2d")

        d = results[0]
        assert len(results) == 1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)

        for i in range(n):
            S.add_data_point(
                dt,
                ("source2d", "randoms"),
                dvalue[i],
                theta=dtheta[i],
                error=derror[i],
                npair=dnpair[i],
                weight=dweight[i],
            )

        self.write_metadata(S, meta)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output("gammat_randoms"), overwrite=True)


# Aperture Mass class that inherits from TXTwoPoint
class TXApertureMass(TXTwoPoint):
    """
    Measure the aperture mass statistics with TreeCorr

    There are real and imaginary components of the aperture mass
    and its cross term.
    """

    name = "TXApertureMass"
    inputs = [
        ("binned_shear_catalog", BinnedCatalog),
        ("shear_photoz_stack", QPNOfZFile),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [
        ("aperture_mass_data", SACCFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = TREECORR_CONFIG | {
        "calcs": StageParameter(
            list, [0, 1, 2], msg="Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position"
        ),
        "source_bins": StageParameter(list, [-1], msg="List of source bins to use (-1 means all)."),
        "lens_bins": StageParameter(list, [-1], msg="List of lens bins to use (-1 means all)."),
        "reduce_randoms_size": StageParameter(float, 1.0, msg="Factor to reduce the size of random catalogs."),
        "var_method": StageParameter(str, "jackknife", msg="Method for computing variance (jackknife, sample, etc.)."),
        "use_true_shear": StageParameter(bool, False, msg="Whether to use true shear values."),
        "subtract_mean_shear": StageParameter(bool, False, msg="Whether to subtract mean shear."),
        "use_randoms": StageParameter(bool, False, msg="Whether to use random catalogs."),
        "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode."),
        "patch_dir": StageParameter(str, "./cache/patches", msg="Directory for storing patch files."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
        "share_patch_files": StageParameter(bool, False, msg="Whether to share patch files across processes."),
    }

    # These two functions can be combined into a single one.
    def _read_nbin_from_tomography(self):
        with self.open_input("binned_shear_catalog") as f:
            nbin_source = f["shear"].attrs["nbin_source"]

        source_list = range(nbin_source)
        lens_list = []  # Not necessary in this subclass

        return source_list, lens_list

    def select_calculations(self, source_list, lens_list):
        # For shear-shear we omit pairs with j>i
        # Lens list is an empty list here, and is unused
        calcs = []
        k = SHEAR_SHEAR
        for i in source_list:
            for j in range(i + 1):
                if j in source_list:
                    calcs.append((i, j, k))

        if self.rank == 0:
            print(f"Running these calculations: {calcs}")

        return calcs

    def calculate_shear_shear(self, i, j):
        gg = super().calculate_shear_shear(i, j)
        gg.mapsq = gg.calculateMapSq()

        return gg

    def write_output(self, source_list, lens_list, meta, results):
        # lens_list is unused in this function, but should always be passed as an empty list
        import sacc

        # Names for aperture-mass correlation functions
        MAPSQ = "galaxy_shear_apmass2"
        MAPSQ_IM = "galaxy_shear_apmass2_im"
        MXSQ = "galaxy_shear_apmass2_cross"
        MXSQ_IM = "galaxy_shear_apmass2_im_cross"

        # Initialise SACC object
        S = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data
        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            for i in source_list:
                z, Nz = f.get_bin_n_of_z(i)
                S.add_tracer("NZ", f"source_{i}", z, Nz)

        # Now build up the collection of data points, adding them all to the sacc
        for d in results:
            # First the tracers and generic tags
            tracer1 = f"source_{d.i}"
            tracer2 = f"source_{d.j}"

            # Skip empty bins
            if d.object is None:
                continue

            theta = np.exp(d.object.meanlogr)
            weight = d.object.weight
            err = np.sqrt(d.object.mapsq[4])
            n = len(theta)
            for j, CORR in enumerate([MAPSQ, MAPSQ_IM, MXSQ, MXSQ_IM]):
                map = d.object.mapsq[j]
                for i in range(n):
                    S.add_data_point(
                        CORR,
                        (tracer1, tracer2),
                        map[i],
                        theta=theta[i],
                        error=err[i],
                        weight=weight[i],
                    )

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        self.write_metadata(S, meta)

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output("aperture_mass_data"), overwrite=True)
