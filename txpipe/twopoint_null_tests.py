from .base_stage import PipelineStage
from .data_types import (
    HDFFile,
    ShearCatalog,
    TomographyCatalog,
    RandomsCatalog,
    SACCFile,
    PNGFile,
    TextFile,
    QPNOfZFile,
    MapsFile
)
import numpy as np
from .twopoint import TXTwoPoint, SHEAR_SHEAR, SHEAR_POS, POS_POS
from .utils import DynamicSplitter


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
        "chunk_rows": 100_000,
        "initial_size": 100_000,
    }

    def run(self):

        cols = ["ra", "dec"]

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

        # Also read the r band mag.  We don't save it as we don't
        # need it later (?) but we do use it for the split
        it = self.iterate_hdf(
            "star_catalog",
            "stars",
            cols + ["mag_r"],
            self.config["chunk_rows"],
        )

        for s, e, data in it:
            print(f"Process 0 binning data in range {s:,} - {e:,}")
            r = data["mag_r"]
            mag_bins = np.repeat("      ", r.size)
            mag_bins = np.zeros(r.size, dtype=int)
            dim_cut = (r > 18.2) & (r < 22.0)
            bright_cut = (r > 14.0) & (r < 18.2)

            dim = {name: col[dim_cut] for name, col in data.items()}
            bright = {name: col[bright_cut] for name, col in data.items()}

            splitter.write_bin(dim, "dim")
            splitter.write_bin(bright, "bright")

        splitter.finish()
        output.close()


class TXGammaTFieldCenters(TXTwoPoint):
    """
    Make diagnostic 2pt measurements of tangential shear around field centers

    This subclass of the standard TXTwoPoint uses the centers
    of exposure fields as "lenses", as a systematics test.
    """

    name = "TXGammaTFieldCenters"
    inputs = [
        ("binned_shear_catalog", ShearCatalog),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
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
    config_options = {
        "calcs": [0, 1, 2],
        "min_sep": 2.5,
        "max_sep": 250,
        "nbins": 20,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "reduce_randoms_size": 1.0,
        "var_method": "shot",
        "npatch": 5,
        "use_true_shear": False,
        "subtract_mean_shear": False,
        "use_randoms": True,
        "patch_dir": "./cache/patches",
        "low_mem": False,
        "chunk_rows": 100_000,
        "share_patch_files": False,
        "use_subsampled_randoms": False,
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

        assert i == 0
        cat = treecorr.Catalog(
            self.get_input("exposures"),
            ext="exposures",
            ra_col="ratel",
            dec_col="dectel",
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

        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            # The last entry represents the 2D n(z)
            z, Nz = f.get_2d_n_of_z(-1)

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer("Misc", "fieldcenter")
        S.add_tracer("NZ", "source2d", z, Nz)

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
        ("binned_shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("random_cats", RandomsCatalog),
        ("binned_star_catalog", HDFFile),
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
    config_options = {
        "calcs": [0, 1, 2],
        "min_sep": 2.5,
        "max_sep": 100,
        "nbins": 20,
        "bin_slop": 1,
        "sep_units": "arcmin",
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "reduce_randoms_size": 1.0,
        "var_method": "shot",
        "npatch": 5,
        "use_true_shear": False,
        "subtract_mean_shear": False,
        "use_randoms": True,
        "patch_dir": "./cache/patches",
        "low_mem": False,
        "chunk_rows": 100_000,
        "share_patch_files": False,
        "use_subsampled_randoms": False,
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

        fig = self.open_output(image_file, wrapper=True)

        # compute the mean and the chi^2/dof
        z = (dvalue) / derror
        chi2 = np.sum(z**2)
        chi2dof = chi2 / (len(dtheta) - 1)

        plt.errorbar(
            dtheta,
            dtheta * dvalue,
            dtheta * derror,
            fmt="ro",
            capsize=3,
            label=rf"$\chi^2/dof = {chi2dof:.2f}$",
        )
        plt.legend(loc="best")
        plt.xscale("log")

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title(f"{text} Star Centers Tangential Shear")

        fig.close()

    def write_output_sacc(self, meta, d, sacc_file, text):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc

        dt = "galaxyStar_shearDensity_xi_t"

        S = sacc.Sacc()

        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            z, Nz = f.get_2d_n_of_z(-1)

        # Add the data points that we have one by one, recording which
        # tracer they each require
        name = "{}_stars".format(text.lower())
        S.add_tracer("Misc", name)
        S.add_tracer("NZ", "source2d", z, Nz)

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
        ("binned_shear_catalog", ShearCatalog),
        ("shear_photoz_stack", QPNOfZFile),
        ("random_cats", RandomsCatalog),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [
        ("gammat_randoms", SACCFile),
        ("gammat_randoms_plot", PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        "calcs": [0, 1, 2],
        "min_sep": 2.5,
        "max_sep": 100,
        "nbins": 20,
        "bin_slop": 1,
        "sep_units": "arcmin",
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "reduce_randoms_size": 1.0,
        "var_method": "shot",
        "npatch": 5,
        "use_true_shear": False,
        "subtract_mean_shear": False,
        "use_randoms": False,
        "patch_dir": "./cache/patches",
        "low_mem": False,
        "chunk_rows": 100_000,
        "share_patch_files": False,
        "use_subsampled_randoms": False,
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

        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            z, Nz = f.get_2d_n_of_z(-1)

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer("Misc", "randoms")
        S.add_tracer("NZ", "source2d", z, Nz)

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
        ("binned_shear_catalog", ShearCatalog),
        ("shear_photoz_stack", QPNOfZFile),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [
        ("aperture_mass_data", SACCFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        "calcs": [0, 1, 2],
        "min_sep": 0.5,
        "max_sep": 300.0,
        "nbins": 15,
        "bin_slop": 0.02,
        "sep_units": "arcmin",
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "source_bins": [-1],
        "lens_bins": [-1],
        "reduce_randoms_size": 1.0,
        "var_method": "jackknife",
        "use_true_shear": False,
        "subtract_mean_shear": False,
        "use_randoms": False,
        "low_mem": False,
        "patch_dir": "./cache/patches",
        "low_mem": False,
        "chunk_rows": 100_000,
        "share_patch_files": False,
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



class TXShearBmode(PipelineStage):
    '''
    Make shear B-mode measurements.
    This stage computes xip and xim in very narrow angular bins and transforms them into 
    Fourier B-modes using narrow window functions. This allows us to compute B-modes without
    being affected by (a) the mask (which is an issue if we use https://arxiv.org/abs/astro-ph/0511629,
    which is implemented in NaMaster, since masks for LSS is more complicated than CMB) and
    (b) noise bias (since the measurements we make here are real-space measurements).
    '''
    name     = "TXShearBmode"
    parallel = False

    inputs  = [
               ("shear_photoz_stack"    , QPNOfZFile), 
               ("source_maps"           , MapsFile),
               ("mask"                  , MapsFile),
               ("twopoint_data_real_raw", SACCFile),
              ]
    
    outputs = [("twopoint_data_fourier_shearbmode", SACCFile),
               ]

    config_options = {
                      'method'    : 'hybrideb', 
                      'purify_b'  : False,
                      'Nell'      : 20,         
                      'lmin'      : 200,         
                      'lmax'      : 2000,       
                      'lspacing'  : 'log',       
                      'bin_file'  : "",              
                      'theta_min' : 2.5,      
                      'theta_max' : 250,      
                      'Ntheta'    : 1000,     
                      'Nsim'      : 1000
                      }
    
    def run(self):
        import os,sys
        import pymaster as nmt
        import healpy as hp
        import sacc
        import pyccl
        import hybrideb
        import datetime
        import pickle
        from tqdm import tqdm

        if self.config["method"] == 'namaster':
            self.run_namaster(self.config["purify_b"])

        elif self.config["method"] == 'hybrideb':
            self.run_hybrideb()

        else:
            raise ValueError("Method must 'namaster' or 'hybrideb'")
            


    def run_namaster(self,purify_b): 
        '''
        B-mode calculation that is already implemented in NaMaster
        '''
        print('running namaster')
        import pymaster as nmt
        import healpy as hp
        from tqdm import tqdm
        import pickle


        lmin  = self.config['lmin']
        lmax  = self.config['lmax']
        Nell  = self.config['Nell']
        Nsims = self.config["Nsims"]
        lspacing = self.config['lspacing']

        if purify_b:
            print("WARNING: Namaster's B-mode purification requires the mask to be heavily apodized.")
            print("For a realistic LSS mask, this will most likely not work, but this function is ")
            print("implemented for completeness nonetheless.")
           
        # Open source maps (g1,g2,weights)
        with self.open_input("source_maps", wrapper=True) as f:

            # Get the number of tomographic bins
            # +1 comes from also loading the non-tomographic sample
            nbin_source = f.file["maps"].attrs["nbin_source"]+1 

            # Load the tomographic samples
            g1_maps     = [f.read_map(f"g1_{b}") for b in range(nbin_source-1)]
            g2_maps     = [f.read_map(f"g2_{b}") for b in range(nbin_source-1)]
            
            # Load non-tomographic sample
            g1_maps.append(f.read_map(f"g1_2D"))
            g2_maps.append(f.read_map(f"g2_2D"))

            if lmax>3*hp.npix2nside(len(g1_maps[0])):
                raise ValueError("lmax must be smaller than 3*nside")

        # Open mask
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")            
        
        # Define binning 
        if lspacing=='lin':
            bine = np.linspace(lmin+1, lmax+1, Nell+1, dtype=np.int64)
        elif lspacing=='log':
            bine = np.geomspace(lmin+1, lmax+1, Nell+1, dtype=np.int64)
        else:
            raise ValueError("lspacing must be either 'log' or 'lin'")
        
        b    = nmt.NmtBin.from_edges(bine[:-1], bine[1:])

        # Initialize the fields
        fields = {}
        for i in range(nbin_source):
            g1_maps[i][g1_maps[i]==hp.UNSEEN]=0
            g2_maps[i][g2_maps[i]==hp.UNSEEN]=0
            fields[i] = nmt.NmtField(mask, [g1_maps[i], g2_maps[i]], purify_e=False, purify_b=purify_b,lmax=lmax)
        
        # To speed the covariance calculation up, one can use multiple nodes to do this calculation externally
        products_dict = {'mask':mask,'Nell':Nell,'nbin_source':nbin_source, 'g1_maps': g1_maps, 'g2_maps': g2_maps}

        file_namaster_intermediate = self.get_output("twopoint_data_fourier_shearbmode")[:-5]+ f"_namaster.pkl"
        
        with open(file_namaster_intermediate, 'wb') as handle:
            pickle.dump(products_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compute Cls and store them in dictionrary
        ret=np.zeros((nbin_source,nbin_source,len(b.get_effective_ells())))

        for zi in range(nbin_source):
            for zj in range(zi,nbin_source):

                if zi!=zj and (zi==nbin_source-1 or zj==nbin_source-1):
                    # No need to take cross-correlation between tomographic and non-tomographic sample
                    continue
                else:    
                    field1 = fields[zi]
                    field2 = fields[zj]
                    w_yp  = nmt.NmtWorkspace()
                    w_yp.compute_coupling_matrix(field1, field2, b)

                    cl_coupled   = nmt.compute_coupled_cell(field1,field2)
                    cl_decoupled = w_yp.decouple_cell(cl_coupled)
                    ret[zj,zi,:]= cl_decoupled[3]
               
        # Compute covariance by randomly rotating shear values in pixels.
        tmparr = np.zeros((int(nbin_source*(nbin_source+1)/2*len(b.get_effective_ells())),Nsims))

        for k in tqdm(range(Nsims)):
            fields = {}
            for i in range(nbin_source):
                # Rotate shear maps randomly to create noise maps
                idx = np.where(g1_maps[i]!=0)[0]
                psi = np.random.uniform(0, 2 * np.pi,size=len(idx))
                g1_maps[i][idx] =  g1_maps[i][idx]*np.cos(2*psi) + g2_maps[i][idx]*np.sin(2*psi)
                g2_maps[i][idx] = -g1_maps[i][idx]*np.sin(2*psi) + g2_maps[i][idx]*np.cos(2*psi)
                
                fields[i] = nmt.NmtField(mask, [g1_maps[i], g2_maps[i]], purify_e=False, purify_b=purify_b, lmax=lmax)

            tmp = np.array([])
            for zi in range(nbin_source):
                for zj in range(zi,nbin_source):
                    
                    field1 = fields[zi]
                    field2 = fields[zj]
                    w_yp  = nmt.NmtWorkspace()
                    w_yp.compute_coupling_matrix(field1, field2, b)
    
                    cl_coupled   = nmt.compute_coupled_cell(field1,field2)
                    cl_decoupled = w_yp.decouple_cell(cl_coupled)
                    tmp = np.concatenate([tmp,cl_decoupled[3]])

            tmparr[:,k] = tmp
            
        n_bins      = b.get_n_bands()
        bin_weights = np.zeros([n_bins, b.lmax+1])
        
        for i in range(n_bins):
            bin_weights[i, b.get_ell_list(i)] = b.get_weight_list(i)

        ell         = b.get_effective_ells()
        bin_weights = bin_weights 
        results     = ret
        cov         = np.cov(tmparr)

        print('Saving ShearBmode Cls in sacc file')   
        self.save_power_spectra(nbin_source, ell, results, cov)
    

    def run_hybrideb(self): 
        '''
        B-mode method of Becker and Rozo 2015 http://arxiv.org/abs/1412.3851
        '''
        import hybrideb
        import pickle
        import sacc
        import os
        
        print('running hybrideb')        
        Nell   = self.config['Nell']
        Nsims  = self.config["Nsims"]
        Ntheta = self.config["Ntheta"]

        # Check if nbins is less than 1000, and throw warning.
        if self.config['Ntheta']<1000:
            print("WARNING: Calculating hybridEB using Ntheta<1000, which may lead to inaccurate results.")

        # Computing the weights takes a few minutes so its a lot faster
        # to precompute them and load them again in subsequent runs.
        file_precomputed_weights = self.get_output("twopoint_data_fourier_pureB")+ f".precomputedweights.pkl"

        if os.path.exists(file_precomputed_weights):
            print(f"{self.rank} WARNING: Using precomputed weights from revious run: {file_precomputed_weights}")
            with open(file_precomputed_weights, "rb") as f:
                geb_dict = pickle.load(f)
        
        else:
            # To run this method we need to first compute the Fourier and real-space windows
            # hybrideb.HybridEB: Sets up the estimator by creating top hat filters in real-space
            # hybrideb.BinEB   : Calculates the intermediate filter quantities, namely fa, fb, M+, M- [Eqs 6-11]
            # hybrideb.GaussEB : Creates Gaussian windows in Fourier space
            heb = hybrideb.HybridEB(self.config['theta_min'], self.config['theta_max'], Ntheta)
            beb = hybrideb.BinEB(self.config['theta_min'], self.config['theta_max'], Ntheta)
            geb = hybrideb.GaussEB(beb, heb)
            
            # geb is in a fromat that can not be naturally saved so we convert it to a regular dictionary.
            geb_dict = {}
            for i in range(0,Nell):
                geb_dict[f"{i+1}"] = {}
                for j in range(0,6):
                    geb_dict[f"{i+1}"][f"{j+1}"]=geb(i)[j]
            
            # geb_dict stores the following information in a regular dictionary format
            # 1: theta in radians
            # 2: Fp  plus component of the real-space window function
            # 3: Fm  minus component of the real-space window function
            # 4: ell
            # 5: Wp  plus component of Fourier-space window function
            # 6: Wm  minus component of Fourier-space window function

            # 2 and 3 are used to get Xp/Xm      = sum((Fp*xip_hat +/- Fm*xim_hat)/2)    [eq1]
            # 5 and 6 are used to get <Xp>/<Xm>  = \int ell factors(ell) (Wp*Pe + Wm*Pb) [eq5]

            # Save this for subsequent runs since the windows don't change.

            with open(file_precomputed_weights, 'wb') as handle:
                pickle.dump(geb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        En0  = np.zeros(Nell)
        Bn0  = np.zeros(Nell)
        En   = np.zeros((Nell,Nsims))
        Bn   = np.zeros((Nell,Nsims))
        ell  = np.zeros(Nell)

        # Load xip/xim that were computed in the TXTwopoint stage and check length required for covariance
        filename   = self.get_input("twopoint_data_real_raw")
        data_twopt = sacc.Sacc.load_fits(filename)
        Qxip       = sacc.standard_types.galaxy_shear_xi_plus

        source_tracers = set()
        for b1, b2 in data_twopt.get_tracer_combinations(Qxip):
            source_tracers.add(b1)
            source_tracers.add(b2)

        nbin_source = max(len(source_tracers), 1)

        corvarr     = np.zeros((int(nbin_source*(nbin_source+1)/2*Nell), Nsims))
        results     = np.zeros((nbin_source,nbin_source,Nell))

        # Loop over all bin combinations
        c = 0
        for zi in range(0,nbin_source):
            for zj in range(zi,nbin_source):
                
                # Initial and final row indices of the resulting array (covarr) to fill.
                # For each tomographic bin we have Nell bins to fill.
                ii = Nell*c
                ff = Nell*(c+1)
                
                tmp  = data_twopt.copy()

                # From the sacc file only load relevant tracer combinations 
                tmp.keep_selection(tracers=(f'source_{zj}', f'source_{zi}'))
                dvec = tmp.mean
                xip  = dvec[:int(Ntheta)]
                xim  = dvec[int(Ntheta):]

                # Make random draws based on mean and covariance of xip/xim measurements
                x = np.random.multivariate_normal(mean = tmp.mean, cov =tmp.covariance.covmat , size = Nsims)
                Rxip = x[:,:int(Ntheta)]
                Rxim = x[:,int(Ntheta):]
                
                # Convert each random draw into B-mode measurement and compute covariance          
                for n in range(Nsims):
                    for i in range(int(Nell)):
                        res     = geb_dict[f"{i+1}"]   
                        Fp      = res['2']
                        Fm      = res['3']
                        En[i,n] = np.sum(Fp*Rxip[n,:] + Fm*Rxim[n,:])/2 
                        Bn[i,n] = np.sum(Fp*Rxip[n,:] - Fm*Rxim[n,:])/2
                        
                corvarr[ii:ff,:] = Bn

                # Compute actual data vector
                for i in range(int(Nell)):
                    res     = geb_dict[f"{i+1}"]   
                    Fp      = res['2']
                    Fm      = res['3']
                    En0[i]   = np.sum(Fp*xip + Fm*xim)/2 
                    Bn0[i]   = np.sum(Fp*xip - Fm*xim)/2
                    ell[i]  = res['4'][np.argmax(res['5'])] # setting ell to the peak of the Gaussian window function

                results[zj,zi,:]= Bn0[:]
                c+=1
                
            cov = np.cov(corvarr)

        print('Saving pureB Cls in sacc file')   
        self.save_power_spectra(nbin_source, ell, results, cov)


    def save_power_spectra(self, nbin_source, ell, results, cov):
        import sacc
        import datetime

        S = sacc.Sacc()
        S.metadata['nbin_source'] = nbin_source
        S.metadata['creation']    = datetime.datetime.now().isoformat()
        S.metadata['method']      = self.config['method']
        S.metadata['info']        = 'ClBB'

        CBB = sacc.standard_types.galaxy_shear_cl_bb

        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            for i in range(nbin_source):
                if i==nbin_source-1:
                    z,Nz  = f.get_2d_n_of_z()
                    S.add_tracer("NZ", f"source_{i}", z, Nz)
                else:
                    z, Nz = f.get_bin_n_of_z(i)
                    S.add_tracer("NZ", f"source_{i}", z, Nz)

        for zi in range(0,nbin_source):
            for zj in range(zi,nbin_source):
                tracer1 = f"source_{zj}"
                tracer2 = f"source_{zi}"
                val     = results[zj,zi,:]
                print(val)
                ell     = ell
                for k in range(0,len(val)):
                    S.add_data_point(CBB, (tracer1, tracer2), value=val[k], ell=ell[k])

        S.add_covariance(cov)
        
        output_filename = self.get_output("twopoint_data_fourier_shearbmode")
        S.save_fits(output_filename, overwrite=True)


