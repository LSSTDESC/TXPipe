from .base_stage import PipelineStage
from .data_types import (
    HDFFile,
    ShearCatalog,
    TomographyCatalog,
    RandomsCatalog,
    SACCFile,
    PNGFile,
    TextFile,
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
        ("shear_photoz_stack", HDFFile),
        ("lens_photoz_stack", HDFFile),
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

        f = self.open_input("shear_photoz_stack")
        z = f["n_of_z/source2d/z"][:]
        Nz = f[f"n_of_z/source2d/bin_0"][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer("misc", "fieldcenter")
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
        ("shear_photoz_stack", HDFFile),
        ("lens_photoz_stack", HDFFile),
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

        f = self.open_input("shear_photoz_stack")
        z = f["n_of_z/source2d/z"][:]
        Nz = f[f"n_of_z/source2d/bin_0"][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        name = "{}_stars".format(text.lower())
        S.add_tracer("misc", name)
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
        ("shear_photoz_stack", HDFFile),
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
        "use_randoms": False,
        "patch_dir": "./cache/patches",
        "low_mem": False,
        "chunk_rows": 100_000,
        "share_patch_files": False,
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

        f = self.open_input("shear_photoz_stack")
        z = f["n_of_z/source2d/z"][:]
        Nz = f[f"n_of_z/source2d/bin_0"][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer("misc", "randoms")
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
        ("shear_photoz_stack", HDFFile),
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
        f = self.open_input("shear_photoz_stack")
        for i in source_list:
            z = f["n_of_z/source/z"][:]
            Nz = f[f"n_of_z/source/bin_{i}"][:]
            S.add_tracer("NZ", f"source_{i}", z, Nz)
        f.close()

        # Now build up the collection of data points, adding them all to the sacc
        for d in results:

            # First the tracers and generic tags
            tracer1 = f"source_{d.i}"
            tracer2 = f"source_{d.j}"

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
