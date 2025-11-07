from .twopoint import TXTwoPoint, TREECORR_CONFIG, SHEAR_POS
from .base_stage import PipelineStage
from .data_types import SACCFile, ShearCatalog, HDFFile, QPNOfZFile, FiducialCosmology, TextFile, PNGFile
import numpy as np
from ceci.config import StageParameter
import os


class TXDeltaSigma(TXTwoPoint):
    """Compute Delta-Sigma, the excess surface density around lenses.

    This is a subclass of TXTwoPoint that re-weights the lenses by the
    inverse critical surface density based on the source redshift distribution.
    """

    name = "TXDeltaSigma"

    inputs = [
        ("binned_shear_catalog", ShearCatalog),
        ("binned_lens_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("tracer_metadata", HDFFile),
        ("patch_centers", TextFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [("delta_sigma", SACCFile)]

    config_options = TREECORR_CONFIG | {
        "source_bins": StageParameter(list, [-1], msg="List of source bins to use (-1 means all)"),
        "lens_bins": StageParameter(list, [-1], msg="List of lens bins to use (-1 means all)"),
        "var_method": StageParameter(str, "jackknife", msg="Method for computing variance (jackknife, sample, etc.)"),
        "use_randoms": StageParameter(bool, True, msg="Whether to use random catalogs"),
        "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode"),
        "patch_dir": StageParameter(str, "./cache/delta-sigma-patches", msg="Directory for storing patch files"),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk when making patches"),
        "share_patch_files": StageParameter(bool, False, msg="Whether to share patch files across processes"),
        "metric": StageParameter(str, "Euclidean", msg="Distance metric to use (Euclidean, Arc, etc.)"),
        "gaussian_sims_factor": StageParameter(
            list,
            default=[1.0],
            msg="Factor by which to decrease lens density to account for increased density contrast.",
        ),
        # "use_subsampled_randoms": StageParameter(bool, False, msg="Use subsampled randoms file for RR calculation"),
        "delta_z": StageParameter(float, 0.001, msg="Z bin width for sigma_crit spline computation"),
    }


    def select_calculations(self, source_list, lens_list):
        calcs = []
        # Fill in these config items that are not relevant for Delta Sigma
        # but are needed by the parent class.
        self.config["do_shear_pos"] = True
        self.config["do_shear_shear"] = False
        self.config["do_pos_pos"] = False
        self.config["use_subsampled_randoms"] = False

        # For Delta Sigma everything is just shear-position
        k = SHEAR_POS
        for i in source_list:
            for j in lens_list:
                calcs.append((i, j, k))

        if self.rank == 0:
            print(f"Running {len(calcs)} calculations: {calcs}")

        return calcs
    

    def read_nbin(self):
        source_list, lens_list = super().read_nbin()
        # This is a convenient place to also compute the sigma_crit_inverse splines
        # that we need later
        self.compute_sigma_crit_inverse_splines(source_list)
        return source_list, lens_list


    def compute_sigma_crit_inverse_spline(self, z, nz):
        import scipy.interpolate
        import scipy.integrate

        # We need a fiducial cosmology to convert redshifts to distances.
        # This is why I don't like Delta Sigma.
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()

        # Another option would be to use the QP object directly and use
        # its .pdf() method. That's probably better than this.
        zs_spline = scipy.interpolate.InterpolatedUnivariateSpline(z, nz, ext="zeros")
        zmax = z.max() + 0.1
        dz = self.config["delta_z"]

        # We integrate this over source redshift for each lens redshift
        # It's just a call to CCL weighted by the n(z)
        def integrand(zl, zs):
            w = np.where(zs > zl)
            a_l = 1 / (1 + zl)
            a_s = 1 / (1 + zs)
            out = np.zeros_like(zs)
            # https://ccl.readthedocs.io/en/latest/api/pyccl.cosmology.html#pyccl.cosmology.Cosmology.sigma_critical
            sigma_crit = cosmo.sigma_critical(a_lens=a_l, a_source=a_s[w])
            n_z = zs_spline(zs[w])
            out[w] = n_z / sigma_crit
            return out

        # Set up the grids needed by the integrals
        zl_grid = np.arange(0, zmax, dz)
        zs_grid = np.arange(0, zmax, dz)
        sigma_crit_inv = np.zeros_like(zl_grid)

        # Do the integrals, just trapezoidal rule
        for i, zli in enumerate(zl_grid[1:-1]):
            integrand_vals = integrand(zli, zs_grid)
            sigma_crit_inv[i] = scipy.integrate.trapezoid(integrand_vals, zs_grid)

        # This gives us the mean sigma_crit_inverse for each lens z
        spline = scipy.interpolate.UnivariateSpline(zl_grid, sigma_crit_inv)
        return spline
    

    def compute_sigma_crit_inverse_splines(self, source_list):
        """
        For each source bin, compute and store the spline
        for sigma_crit_inverse as a function of lens redshift.
        """
        from .utils import blank
        splines = {}
        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            for i in source_list:
                if i == "all":
                    z, Nz = f.get_2d_n_of_z()
                else:
                    z, Nz = f.get_bin_n_of_z(i)
                if self.rank == 0:
                    print("Computing sigma_crit_inverse spline for source bin", i)
                splines[i] = self.compute_sigma_crit_inverse_spline(z, Nz)
        blank.sigma_crit_inverse_splines = splines

    def get_shear_catalog(self, i):
        """
        This is a total hack. We want the get_lens_catalog to re-scale
        the weights based on which source bin is being used. So we store
        the source bin index in self.active_shear_catalog here, so that
        get_lens_catalog can access it.

        This only works because the parent class calls get_shear_catalog
        before get_lens_catalog.

        If that does change it should be obvious because self.active_shear_catalog
        will be undefined.
        """
        self.active_shear_catalog = i
        return super().get_shear_catalog(i)

 
    def get_lens_catalog(self, i):
        import treecorr
        from .utils import blank
        treecorr.Catalog.eval_modules.append("txpipe.utils.blank as splines")

        # To get the scaling into Treecorr without re-creating multiple
        # copies of all the files we have to pass the redshift into
        # another module that treecorr can access and use the "w_eval"
        # parameter to tell it to do it.
        with self.open_input("binned_lens_catalog") as f:
            redshift = f[f"/lens/bin_{i}/z"][:]
        blank.redshift = redshift
        
        # Load the catalog. Note that we are not loading the weight
        # from the file, because we want to scale it by 1/sigma_crit
        # here.
        cat = treecorr.Catalog(
            self.get_input("binned_lens_catalog"),
            ext=f"/lens/bin_{i}",
            ra_col="ra",
            dec_col="dec",
            w_col="weight",
            ra_units="degree",
            dec_units="degree",
            # Apply a function that rescales the weights
            w_eval=f"weight / splines.sigma_crit_inverse_splines[{self.active_shear_catalog}](splines.redshift)",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_lens_catalog", i),
        )

        return cat
    
    def write_output(self, source_list, lens_list, meta, results):
        import sacc
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()

        # Get the source n(z) which are just passed into the output file
        S = sacc.Sacc()
        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            for i in source_list:
                z, Nz = f.get_bin_n_of_z(i)
                S.add_tracer("NZ", f"source_{i}", z, Nz)

        # Get the lens n(z), from which we also need the mean z
        # so that we can turn angles into physical radii.
        with self.open_input("lens_photoz_stack", wrapper=True) as f:
            # For both source and lens
            qp_object = f.read_ensemble()

            # we skip the "all" bin for now as we are not running it anyway
            lens_mean_z = qp_object.mean()[:-1]
            for i in lens_list:
                z, Nz = f.get_bin_n_of_z(i)
                S.add_tracer("NZ", f"lens_{i}", z, Nz)

        # Loop through calculated results and add to SACC
        for d in results:
            tracer1 = f"source_{d.i}"
            tracer2 = f"lens_{d.j}"

            xi = d.object.xi
            theta = np.exp(d.object.meanlogr)
            # Convert theta to a radius in physical units at the lens redshift
            a_lens = 1 / (1 + np.mean(lens_mean_z[d.j]))
            r_mpc = cosmo.angular_diameter_distance(a_lens) * np.radians(theta / 60)

            weight = d.object.weight
            err = np.sqrt(d.object.varxi)
            n = len(xi)
            for i in range(n):
                S.add_data_point(
                    "galaxy_shearDensity_deltasigma",
                    (tracer1, tracer2),
                    xi[i],
                    theta=theta[i],
                    r_mpc=r_mpc[i],
                    error=err[i],
                    weight=weight[i],
                )

        # add a few bits of handy metadata
        meta["nbin_source"] = len(source_list)
        meta["nbin_lens"] = len(lens_list)
        self.write_metadata(S, meta)
        S.save_fits(self.get_output("delta_sigma"), overwrite=True)


class TXDeltaSigmaPlots(PipelineStage):
    """Make plots of Delta Sigma results.
    
    """
    name = "TXDeltaSigmaPlots"
    inputs = [
        ("delta_sigma", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),
        
    ]
    outputs = [
        ("delta_sigma_plot", PNGFile),
        ("delta_sigma_r_plot", PNGFile),
]
    config_options = {}

    def run(self):
        import sacc
        import matplotlib.pyplot as plt

        sacc_data = sacc.Sacc.load_fits(self.get_input("delta_sigma"))

        # Plot in theta coordinates
        nbin_source = sacc_data.metadata['nbin_source']
        nbin_lens = sacc_data.metadata['nbin_lens']
        with self.open_output("delta_sigma_plot", wrapper=True, figsize=(5*nbin_lens, 4*nbin_source)) as fig:
            axes = fig.file.subplots(nbin_source, nbin_lens, squeeze=False)
            for s in range(nbin_source):
                for l in range(nbin_lens):
                    axes[s, l].set_title(f"Source {s}, Lens {l}")
                    axes[s, l].set_xlabel("Radius [arcmin]")
                    axes[s, l].set_ylabel(r"$\Delta \Sigma [M_\odot / pc^2]")
                    axes[s, l].grid()
                    x = sacc_data.get_tag("theta", tracers=(f"source_{s}", f"lens_{l}"))
                    y = sacc_data.get_mean(tracers=(f"source_{s}", f"lens_{l}"))
                    axes[s, l].plot(x, y)
            plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Plot in r coordinates
        nbin_source = sacc_data.metadata['nbin_source']
        nbin_lens = sacc_data.metadata['nbin_lens']
        with self.open_output("delta_sigma_r_plot", wrapper=True, figsize=(5*nbin_lens, 4*nbin_source)) as fig:
            axes = fig.file.subplots(nbin_source, nbin_lens, squeeze=False)
            for s in range(nbin_source):
                for l in range(nbin_lens):
                    axes[s, l].set_title(f"Source {s}, Lens {l}")
                    axes[s, l].set_xlabel("Radius [Mpc]")
                    axes[s, l].set_ylabel(r"$R \cdot \Delta \Sigma [M_\odot / pc^2]$")
                    axes[s, l].grid()
                    x = sacc_data.get_tag("r_mpc", tracers=(f"source_{s}", f"lens_{l}"))
                    y = sacc_data.get_mean(tracers=(f"source_{s}", f"lens_{l}"))
                    axes[s, l].plot(x, y * np.array(x))
            plt.subplots_adjust(hspace=0.3, wspace=0.3)