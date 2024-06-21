from ..base_stage import PipelineStage
from ..twopoint import TXTwoPoint
from ..data_types import (
    HDFFile,
    ShearCatalog,
    SACCFile,
    TextFile,
    MapsFile,
    QPNOfZFile,
    FiducialCosmology,
)
from ..utils.patches import PatchMaker
import numpy as np
import collections
import sys
import os
import pathlib
from time import perf_counter
import gc
from ..utils import choose_pixelization

# This creates a little mini-type, like a struct,
# for holding individual measurements
Measurement = collections.namedtuple("Measurement", ["corr_type", "object", "i", "j"])

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

#external cross correlations
POS_EXT = 3
SHEAR_EXT = 4

#Selfcalibration of IA correlations
SHEAR_SOURCE = 5
SHEAR_SOURCE_SEL = 6
SOURCE_SOURCE = 7

class TXTwoPointSelfCalibrationIA(TXTwoPoint):
    """
    This is the class to calculate the 2pt measurements needed for doing
    self calibration of Intrinsic alignment. This is done with TreeCorr.

    This stage will make the measurements for galaxy-galaxy lensing in
    source bins, and the same measurements imposing a selection function
    """
    name = "TXTwoPointSCIA"
    inputs = [
        ('binned_shear_catalog', ShearCatalog),
        ('binned_random_catalog_source', HDFFile),
        ('shear_photoz_stack', QPNOfZFile),
        ('patch_centers', TextFile),
        ('fiducial_cosmology', FiducialCosmology),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('twopoint_data_SCIA', SACCFile),
        ('twopoint_gamma_x_SCIA', SACCFile),
    ]
    config_options = {
        "calcs": [5,6,7], #IS THIS LINE STILL NEEEDED?
        "min_sep": 2.5,
        "max_sep": 250.0,
        "nbins": 20,
        "bin_slop": 0.0,
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "source_bins": [-1],
        "lens_bins": [-1],
        "reduce_randoms_size": 1.0,
        "do_shear_source": True,
        "do_shear_source_select": True,
        "do_source_source": False,
        "var_method": "jackknife",
        "use_randoms": False,
        "low_mem": False,
        "patch_dir": "./cache/pathces",
        "chunk_row": 100_000,
        "share_patch_files": False,
        "metric": "Rperp",
        "3Dcoords": True,
        "gaussian_sims_factor": [1.],
        "use_subsampled_randoms": True,
    }

    def run(self):
        """
        run method
        """
        import sacc
        import healpy
        import treecorr
        # Binning information
        source_list, lens_list = self.read_nbin()

        if self.rank == 0:
            metric = self.config["metric"] if "metric" in self.config else "Rperp"
            print(f"Running TreeCorr with metric \"{metric}\"")

        meta = self.read_metadata()

        calcs = self.select_calculations(source_list, lens_list)
        sys.stdout.flush()

        self.prepare_patches(calcs, meta)

        results = []
        for i, j, k in calcs:
            result = self.call_treecorr(i, j, k)
            results.append(result)

        if self.comm:
            self.comm.Barrier()

        self.write_output(source_list, lens_list, meta, results)
    
    def _read_nbin_from_tomography(self):
        if self.get_input("binned_shear_catalog") == "none":
            nbin_source = 0
        else:
            with self.open_input("binned_shear_catalog") as f:
                nbin_source = f["shear"].attrs["nbin_source"]
        
        nbin_lens = 0

        source_list = list(range(nbin_source))
        lens_list = list(range(nbin_lens))

        return source_list, lens_list

    def select_calculations(self, source_list, lens_list):
        calcs = []

        if self.config["do_shear_source"]:
            k = SHEAR_SOURCE
            for i in source_list:
                calcs.append((i,i,k))
    
        if self.config["do_shear_source_select"]:
            k = SHEAR_SOURCE_SEL
            for i in source_list:
                calcs.append((i,i,k))
    
        if self.config["do_source_source"]:
            if not self.config["use_randoms"]:
                raise ValueError(
                "You need to have a random catalog to calculate the source-source correlations"
                )
            k = SOURCE_SOURCE
            for i in source_list:
                calcs.append((i,i,k))
    
        if self.rank == 0:
            print(f"Running {len(calcs)} calculations: {calcs}")
    
        return calcs
  
    def prepare_patches(self, calcs, meta):
        """
        For each catalog to be generated, have one process load the catalog
        and write its patch files out to disc.  These are then re-used later
        by all the different processes.

        Parameters
        ----------

        calcs: list
            A list of (bin1, bin2, bin_type) where bin1 and bin2 are indices
            or bin labels and bin_type is one of the constants SHEAR_SHEAR,
            SHEAR_POS, or POS_POS.

        meta: dict
            A dict to which the number of patches (or zero, if no patches) will
            be added for each catalog type, with keys "npatch_shear", "npatch_pos",
            and "npatch_ran".
        """
        cats = set()

        for i, j, k in calcs:
            if k == SHEAR_SOURCE:
                cats.add((i, SHEAR_SHEAR))
                cats.add((j, SHEAR_SHEAR)) # should in principle be changed to SOURCE_SOURCE
            elif k == SHEAR_SOURCE_SEL:
                cats.add((i, SHEAR_SHEAR))
                cats.add((j, SHEAR_SHEAR)) # same as above.
            elif k == SOURCE_SOURCE:
                cats.add((i, SOURCE_SOURCE))
                cats.add((j, SOURCE_SOURCE))
        cats = list(cats)
        cats.sort(key=str)

        chunk_rows = self.config["chunk_rows"]
        npatch_shear = 0
        npatch_pos = 0
        npatch_ran = 0

        self.empty_patch_exists = {}

        for (h, k) in cats:
            ktxt = "shear" if k == SHEAR_SHEAR else "position"
            print(f"Rank {self.rank} making patches for {ktxt} catalog bin {h}")

            if k == SHEAR_SHEAR:
                cat = self.get_shear_catalog(h)
                npatch_shear, contains_empty = PatchMaker.run(cat, chunk_rows, self.comm)
                self.empty_patch_exists[cat.save_patch_dir] = contains_empty
                del cat
            else:
                cat = self.get_shear_catalog(h)
                npatch_pos, contains_empty = PatchMaker.run(cat, chunk_rows, self.comm)
                self.empty_patch_exists[cat.save_patch_dir] = contains_empty
                del cat

                ran_cat = self.get_random_catalg(h)

                if ran_cat is None:
                    continue
                npatch_ran, contains_empty = PatchMaker.run(ran_cat, chunk_rows, self.comm)
                self.empty_patch_exists[ran_cat.save_patch_dir] = contains_empty
                del ran_cat

                if self.config["use_subsampled_randoms"]:
                    ran_cat = self.get_subsampled_random_catalog(h)
                    npatch_ran, contains_empty = PatchMaker.run(ran_cat, chunk_rows, self.comm)
                    self.empty_patch_exists[ran_cat.save_patch_dir] = contains_empty
                    del ran_cat

        meta["npatch_shear"] = npatch_shear
        meta["npatch_pos"] = npatch_pos
        meta["npatch_ran"] = npatch_ran

        if self.comm is not None:
            self.comm.Barrier()

    def call_treecorr(self, i, j, k):
        """
        The all important treecorr wrapper
        """
        import sacc
        import pickle

        pickle_filename = self.get_output("twopoint_data_SCIA") + f".checkpoint-{i}-{j}-{k}.pkl"

        if os.path.exists(pickle_filename):
            print(f"{self.rank} WARNING USING THIS PICKLE FILE I FOUND: {pickle_filename}")
            with open(pickle_filename, "rb") as f:
                result = pickle.load(f)
            return result

        if k == SHEAR_SOURCE:
            xx = self.calculate_shear_pos(i,j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k == SHEAR_SOURCE_SEL:
            xx = self.calculate_shear_pos_select(i, j)
            xtype = sacc.build_data_type_name('galaxy', ['shear', 'Density'], 'xi', subtype = 'ts')
        elif k == SOURCE_SOURCE:
            xx = self.calculate_pos_pos(i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        else:
            raise ValueError(f"Unknown correlation function {k}")

        gc.collect()

        result = Measurement(xtype, xx, i, j)

        sys.stdout.flush()

        if self.comm:
            self.comm.Barrier()

        if self.rank == 0:
            print(f"Pickling result to {pickle_filename}")
            with open(pickle_filename, "wb") as f:
                pickle.dump(result, f)

        return result

    def calculate_shear_pos(self, i, j):
        import treecorr
        import pyccl as ccl

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        n_i = cat_i.nobj

        cat_j = self.get_shear_catalog(j)
        cat_j = self.touch_patches(cat_j)
        rancat_j = self.get_random_catalog(j)
        rancat_j = self.touch_patches(rancat_j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        # We will calculate the separation in Mpc that corresponds to min_sep and max_sep, as if these were given in arcminutes!
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl()
        r_mean_i = np.mean(cat_i.r)
        a_i = ccl.scale_factor_of_chi(cosmo, r_mean_i)
        Da_i = ccl.angular_diameter_distance(cosmo, 1, a2=a_i)
        config = self.config.copy()
        config['min_sep'] = self.config['min_sep']*np.pi*Da_i / 10_800
        config['max_sep'] = self.config['max_sep']*np.pi*Da_i / 10_800

        if self.rank == 0:
            print(f"Calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")
            print(config)

        if n_i == 0 or n_j == 0:
            if self.rank == 0:
                print("Empty catalog: returning None")
                return None

        ng = treecorr.NGCorrelation(config)
        t1 = perf_counter()
        ng.process(cat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])

        if rancat_j:
            rg = treecorr.NGCorrelation(config)
            rg.process(rancat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])
        else:
            rg = None

        ng.calculateXi(rg=rg)
        t2 = perf_counter()
        if self.rank == 0:
            print(f"Processing took {t2 - t1:.1f} seconds")

        return ng

    def calculate_shear_pos_select(self, i, j):
        import treecorr
        import pyccl as ccl

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        n_i = cat_i.nobj

        cat_j = self.get_shear_catalog(j)
        cat_j = self.touch_patches(cat_j)
        rancat_j = self.get_random_catalog(j)
        rancat_j = self.touch_patches(rancat_j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        # We will calculate the separation in Mpc that corresponds to min_sep and max_sep, as if these were given in arcminutes!
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl()
        r_mean_i = np.mean(cat_i.r)
        a_i = ccl.scale_factor_of_chi(cosmo, r_mean_i)
        Da_i = ccl.angular_diameter_distance(cosmo, 1, a2=a_i)
        config = self.config.copy()
        config['min_sep'] = self.config['min_sep']*np.pi*Da_i / 10_800
        config['max_sep'] = self.config['max_sep']*np.pi*Da_i / 10_800

        if self.rank == 0:
            print(f"Calculating shear-position selected bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")
            print(config)

        if n_i == 0 or n_j == 0:
            if self.rank == 0:
                print("Empty catalog: returning None")
                return None

        ng = treecorr.NGCorrelation(config, max_rpar=0.0)
        t1 = perf_counter()
        ng.process(cat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])

        if rancat_j:
            rg = treecorr.NGCorrelation(config, max_rpar=0.0)
            rg.process(rancat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])
        else:
            rg = None

        ng.calculateXi(rg=rg)
        t2 = perf_counter()
        if self.rank == 0:
            print(f"Processing took {t2 - t1:.1f} seconds")

        return ng

    def calculate_pos_pos(self, i, j):
        import treecorr
        import pyccl as ccl

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        rancat_i = self.get_random_catalog(i)
        rancat_i = self.touch_patches(rancat_i)
        n_i = cat_i.nobj
        n_rand_i = rancat_i.nobj if rancat_i is not None else 0

        if i == j:
            cat_j = None
            rancat_j = rancat_i
            n_j = n_i
            n_rand_j = n_rand_i
        else:
            cat_j = self.get_shear_catalog(j)
            cat_j = self.touch_patches(cat_j)
            rancat_j = self.get_random_catalog(j)
            rancat_j = self.touch_patches(rancat_j)
            n_j = cat_j.nobj
            n_rand_j = rancat_j.nobj

        if self.config['use_subsampled_randoms']:
            rancat_sub_i = self.get_subsampled_random_catalog(i)
            rancat_sub_i = self.touch_patches(rancat_sub_i)
            n_rand_sub_i = rancat_sub_i.nobj if rancat_sub_i is not None else 0

            if i == j:
                rancat_sub_j = rancat_sub_i
                n_rand_sub_j = n_rand_sub_i
            else:
                rancat_sub_j = self.get_subsampled_random_catalog(j)
                rancat_sub_j = self.touch_patches(rancat_sub_j)
                n_rand_sub_j = rancat_sub_j.nobj if rancat_sub_j is not None else 0

        if self.rank == 0:
            print(
                f"Calculating source-source bin pair ({i}, {j}): {n_i} x {n_j} objects,  {n_rand_i} x {n_rand_j} randoms"
            )
            if self.config["use_subsampled_randoms"]:
                print(f"and for the rr term, {n_rand_sub_i} x {n_rand_sub_j} pairs")

        if n_i == 0 or n_j == 0:
            if self.rank == 0:
                print("Empty catalog: returning None")
            return None

        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl()
        r_mean_i = np.mean(cat_i.r)
        a_i = ccl.scale_factor_of_chi(cosmo, r_mean_i)
        Da_i = ccl.angular_diameter_distance(cosmo, 1, a2=a_i)
        config = self.config.copy()
        config['min_sep'] = self.config['min_sep']*np.pi*Da_i / 10_800
        config['max_sep'] = self.config['max_sep']*np.pi*Da_i / 10_800

        t1 = perf_counter()

        nn = treecorr.NNCorrelation(config)
        nn.process(cat_i, cat_j, comm=self.comm, low_mem=self.config["low_mem"])

        nr = treecorr.NNCorrelation(config)
        nr.process(cat_i, rancat_j, comm=self.comm, low_mem=self.config["low_mem"])

        if i == j:
            rancat_j = None
            rancat_sub_j = None

        rr = treecorr.NNCorrelation(config)
        if self.confif["use_subsampled_randoms"]:
            rr.process(rancat_sub_i, rancat_sub_j, comm=self.comm, low_mem=self.config["low_mem"])
        else:
            rr.process(rancat_i, rancat_j, comm=self.comm, low_mem=self.config["low_mem"])

        if i == j:
            rn = None
        else:
            rn = treecorr.NNCorrelation(config)
            rn.process(rancat_i, cat_j, comm=self.comm, low_mem=self.config["low_mem"])

        t2 = perf_counter()
        nn.calculateXi(rr, dr=nr, rd=rn)
        if self.rank == 0:
            print(f"Processing took {t2-t1:.1f} seconds")

        return nn
    
    def get_shear_catalog(self, i):
        import treecorr

        cat = treecorr.Catalog(
            self.get_input("binned_shear_catalog"),
            ext=f"/shear/bin_{i}",
            g1_col="g1",
            g2_col="g2",
            r_col="r",
            ra_col="ra",
            dec_col="dec",
            w_col="weight",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_shear_catalog", i),
            flip_g1=self.config["flip_g1"],
            flip_g2=self.config["flip_g2"],
        )

        return cat

    def get_random_catalog(self, i):
        import treecorr

        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("binned_random_catalog_source"),
            ext=f"/randoms/bin_{i}",
            ra_col="ra",
            dec_col="dec",
            r_col="r",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_random_catalog_source", i),
        )

        return rancat
    
    def get_subsampled_random_catalog(self, i):
        import treecorr

        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("binned_random_catalog_source_sub"),
            ext=f"/randoms/bin_{i}",
            ra_col="ra",
            dec_col="dec",
            r_col="r",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_random_catalog_source_sub", i),
        )

        return rancat

    def write_output(self, source_list, lens_list, meta, results):
        import sacc
        import treecorr

        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        WTHETA = sacc.standard_types.galaxy_density_xi
        # We define a new sacc data type for our selection function results.
        GAMMATS = sacc.build_data_type_name(
            "galaxy", ["shear", "Density"], "xi", subtype="ts"
        )
        GAMMAXS = sacc.build_data_type_name(
            "galaxy", ["shear", "Density"], "xi", subtype="xs"
        )

        S = sacc.Sacc()
        S2 = sacc.Sacc()

        if source_list:
            with self.open_input("shear_photoz_stack", wrapper=True) as f:
                for i in source_list:
                    z, Nz = f.get_bin_n_of_z(i)
                    S.add_tracer("NZ", f"source_{i}", z, Nz)
                    S2.add_tracer("NZ", f"source_{i}", z, Nz)


        # Now build up the collection of data points, adding them all to
        # the sacc data one by one.
        self.add_data_points(S, results)

        # Adding the gammaX calculation:
        self.add_gamma_x_data_points(S2, results)

        # The other processes are only needed for the covariance estimation.
        # They do a bunch of other stuff here that isn't actually needed, but
        # it should all be very fast. After this point they are not needed
        # at all so return
        if self.rank != 0:
            return

        S.to_canonical_order()

        self.write_metadata(S, meta)

        S.save_fits(self.get_output("twopoint_data_SCIA"), overwrite=True)

        S2.to_canonical_order()
        self.write_metadata(S2, meta)

        S2.save_fits(self.get_output('twopoint_gamma_x_SCIA'), overwrite=True)

    def add_data_points(self, S, results):
        import treecorr
        import sacc

        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        WTHETA = sacc.standard_types.galaxy_density_xi
        # We define a new sacc data type for our selection function results.
        GAMMATS = sacc.build_data_type_name(
            "galaxy", ["shear", "Density"], "xi", subtype="ts"
        )
        GAMMAXS = sacc.build_data_type_name(
            "galaxy", ["shear", "Density"], "xi", subtype="xs"
        )

        comb = []
        for index, d in enumerate(results):
            tracer1 = f"source_{d.i}"
            tracer2 = f"source_{d.j}"

            if d.object is None:
                continue

            theta = np.exp(d.object.meanlogr)
            npair = d.object.npairs
            weight = d.object.weight

            if d.i == d.j:
                npair = npair / 2
                weight = weight / 2

            xi = d.object.xi
            err = np.sqrt(d.object.varxi)
            n = len(xi)
            for i in range(n):
                S.add_data_point(
                    d.corr_type,
                    (tracer1, tracer2),
                    xi[i],
                    theta=theta[i],
                    error=err[i],
                    npair=npair[i],
                    weight=weight[i],
                )
            comb.append(d.object)

        if treecorr.__version__.startswith("4.2."):
            if self.rank == 0:
                print("Using old TreeCorr - covariance may be slow. "
                      "Consider using 4.3 from github main branch.")
            cov = treecorr.estimate_multi_cov(comb, self.config["var_method"])
        else:
            if self.rank == 0:
                print("Using new TreeCorr 4.3 or above")
            cov = treecorr.estimate_multi_cov(comb, self.config["var_method"], comm=self.comm)
        S.add_covariance(cov)

    def add_gamma_x_data_points(self, S, results):
        import treecorr
        import sacc

        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        GAMMATS = sacc.build_data_type_name(
            "galaxy", ["shear", "Density"], "xi", subtype="ts"
        )
        GAMMAXS = sacc.build_data_type_name(
            "galaxy", ["shear", "Density"], "xi", subtype="xs"
        )
        covs = []
        for index, d in enumerate(results):
            tracer1 = f"source_{d.i}"
            tracer2 = f"source_{d.j}"

            if d.corr_type == GAMMAT:
                theta = np.exp(d.object.meanlogr)
                npair = d.object.npairs
                weight = d.object.weight
                xi_x = d.object.xi_im
                covX = d.object.estimate_cov("shot")
                covs.append(covX)
                err = np.sqrt(np.diag(covX))
                n = len(xi_x)
                for i in range(n):
                    S.add_data_point(
                        GAMMAX,
                        (tracer1, tracer2),
                        xi_x[i],
                        theta=theta[i],
                        error=err[i],
                        weight=weight[i],
                    )
            if d.corr_type == GAMMATS:
                theta = np.exp(d.object.meanlogr)
                npair = d.object.npairs
                weight = d.object.weight
                xi_x = d.object.xi_im
                covX = d.object.estimate_cov("shot")
                covs.append(covX)
                err = np.sqrt(np.diag(covX))
                n = len(xi_x)
                for i in range(n):
                    S.add_data_point(
                        GAMMAXS,
                        (tracer1, tracer2),
                        xi_x[i],
                        theta=theta[i],
                        error=err[i],
                        weight=weight[i],
                    )
        S.add_covariance(covs)

class TXTwoPointSourcePixels(TXTwoPointSelfCalibrationIA):
    """
    This is utilizing the interface for source only implemented above to 
    look at calculations with sources, using pixel maps instead of large
    catalogs. 
    """
    name = "TXTwoPointSourcePixel"
    inputs = [
        ("source_maps", MapsFile),
        ("binned_shear_catalog", ShearCatalog),
        ("binned_random_catalog", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
        ("mask", MapsFile),
    ]
    outputs = [("twopoint_data_source_real_raw", SACCFile), ("twopoint_source_gamma_x", SACCFile)]
    # Add values to the config file that are not previously defined
    config_options = {
        # TODO: Allow more fine-grained selection of 2pt subsets to compute
        "calcs": [0, 1, 2],
        "min_sep": 0.5,
        "max_sep": 300.0,
        "nbins": 9,
        "bin_slop": 0.0,
        "sep_units": "arcmin",
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "source_bins": [-1],
        "lens_bins": [-1],
        "reduce_randoms_size": 1.0,
        "do_shear_shear": True,
        "do_shear_pos": True,
        "do_pos_pos": True,
        "var_method": "jackknife",
        "low_mem": False,
        "patch_dir": "./cache/patches",
        "chunk_rows": 100_000,
        "share_patch_files": False,
        "metric": "Euclidean",
        "use_randoms": True,
        "auto_only": False,
        "gaussian_sims_factor": [1.], 
        "use_subsampled_randoms":False, #not used for pixel estimator,
        "use_sources_only": False,
    }

    def select_calculations(self, source_list, lens_list):
        calcs = []

        if self.config["do_source_source"]:
            k=SOURCE_SOURCE
            if self.config["auto_only"]:
                for i in source_list:
                    calcs.append((i,i,k))
            else:
                for i in source_list:
                    for j in range(i+1):
                        if j in source_list:
                            calcs.append((i,j,k))
        
        if self.config["do_shear_source"]:
            k = SHEAR_SOURCE
            for i in source_list:
                for j in range(i+1):
                    if j in source_list:
                        calcs.append((i,j,k))

        if self.rank == 0:
            print(f"Running {len(calcs)} calculations: {calcs}")

        return calcs
    

    def get_density_map(self, i):
        import treecorr
        
        with self.open_input("source_maps", wrapper=True) as f:
            info = f.read_map_info(f"count_{i}")
            map_d, pix, nside = f.read_healpix(f"count_{i}", return_all=True)
            map_g1, pix_g1, nside  = f.read_healpix(f"g1_{i}", return_all=True)
            map_g2, pix_g2, nside = f.read_healpix(f"g2_{i}", return_all=True)
            print(f"Loaded {i} source maps")

            mask_unseen = (map_g1[pix_g1]>-1e30)*(map_g2[pix_g2]>-1e30)

            # Read the mask to get fracdet weights
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")

        scheme = choose_pixelization(**info) 
        ra_pix, dec_pix = scheme.pix2ang(pix)

        cat = treecorr.Catalog(
            ra=ra_pix[mask_unseen],
            dec=dec_pix[mask_unseen],
            w=mask[pix][mask_unseen],
            k=map_d[pix][mask_unseen],
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
        )

        return cat


    def get_shear_map(self, i):
        import treecorr
        import pdb
        
        with self.open_input("source_maps", wrapper=True) as f:
            info_g1 = f.read_map_info(f"g1_{i}")
            map_g1, pix_g1, nside  = f.read_healpix(f"g1_{i}", return_all=True)
            print(f"Loaded shear 1 {i} maps")

            info_g2 = f.read_map_info(f"g2_{i}")
            map_g2, pix_g2, nside = f.read_healpix(f"g2_{i}", return_all=True)
            print(f"Loaded shear 2 {i} maps")

        scheme = choose_pixelization(**info_g1)
        ra_pix, dec_pix = scheme.pix2ang(pix_g1)

        mask_unseen = (map_g1[pix_g1]>-1e30)*(map_g2[pix_g2]>-1e30)

        cat = treecorr.Catalog(
            ra=ra_pix[mask_unseen],
            dec=dec_pix[mask_unseen],
            #w=,
            g1=map_g1[pix_g1][mask_unseen],
            g2=map_g2[pix_g2][mask_unseen],
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            flip_g1=self.config["flip_g1"],
            flip_g2=self.config["flip_g2"],
            
        )
        return cat

    
    def calculate_shear_pos(self, i, j):
        import treecorr

        cat_i = self.get_shear_map(i)

        cat_j = self.get_density_map(j)

        if self.rank == 0:
            print(
                f"Calculating shear-position bin pair ({i},{j})."
            )

        kg = treecorr.KGCorrelation(self.config)
        t1 = perf_counter()
        kg.process(cat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])

        return kg

    
    def calculate_pos_pos(self, i, j):
        import treecorr

        cat_i = self.get_density_map(i)


        if i == j:
            cat_j = cat_i
        else:
            cat_j = self.get_density_map(j)

        if self.rank == 0:
            print(
                f"Calculating position-position bin pair ({i}, {j})"
            )

        t1 = perf_counter()

        kk = treecorr.KKCorrelation(self.config)
        kk.process(cat_i, cat_j, comm=self.comm, low_mem=self.config["low_mem"])

        return kk
    

class TXTwoPointSCIAArc(TXTwoPointSelfCalibrationIA):
    """
    This is an experimental class, for calculating the self-calibration terms but using 
    the "Arc" metric and thereby bypassing the conversions needed in it's parent class.
    """
    name = "TXTwoPointSCIAArc"
    inputs = [
        ('binned_shear_catalog', ShearCatalog),
        ('binned_random_catalog_source', HDFFile),
        ('shear_photoz_stack', QPNOfZFile),
        ('patch_centers', TextFile),
        ('fiducial_cosmology', FiducialCosmology),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('twopoint_data_SCIA', SACCFile),
        ('twopoint_gamma_x_SCIA', SACCFile),
    ]

    config_options = {
        "calcs": [5,6,7], #IS THIS LINE STILL NEEEDED?
        "min_sep": 2.5,
        "max_sep": 250.0,
        "nbins": 20,
        "bin_slop": 0.0,
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "source_bins": [-1],
        "lens_bins": [-1],
        "reduce_randoms_size": 1.0,
        "do_shear_source": True,
        "do_shear_source_select": True,
        "do_source_source": False,
        "var_method": "jackknife",
        "use_randoms": False,
        "low_mem": False,
        "patch_dir": "./cache/pathces",
        "chunk_row": 100_000,
        "share_patch_files": False,
        "metric": "Arc",
        "sep_units": "arcmin",
        "3Dcoords": True,
        "gaussian_sims_factor": [1.],
        "use_subsampled_randoms": True
    }

    def get_shear_catalog(self, i):
        import treecorr

        cat = treecorr.Catalog(
            self.get_input("binned_shear_catalog"),
            ext=f"/shear/bin_{i}",
            g1_col="g1",
            g2_col="g2",
            r_col="z", #Note we are trying to load in the actual redshift as our r coordinate. 
            ra_col="ra",
            dec_col="dec",
            w_col="weight",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_shear_catalog", i),
            flip_g1=self.config["flip_g1"],
            flip_g2=self.config["flip_g2"],
        )

        return cat

    def calculate_shear_pos(self, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        n_i = cat_i.nobj

        cat_j = self.get_shear_catalog(j)
        cat_j = self.touch_patches(cat_j)
        rancat_j = self.get_random_catalog(j)
        rancat_j = self.touch_patches(rancat_j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        if self.rank == 0:
            print(f"Calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")

        if n_i == 0 or n_j == 0:
            if self.rank == 0:
                print("Empty catalog: returning None")
                return None

        ng = treecorr.NGCorrelation(self.config)
        t1 = perf_counter()
        ng.process(cat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])

        if rancat_j:
            rg = treecorr.NGCorrelation(self.config)
            rg.process(rancat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])
        else:
            rg = None

        ng.calculateXi(rg=rg)
        t2 = perf_counter()
        if self.rank == 0:
            print(f"Processing took {t2 - t1:.1f} seconds")

        return ng

    def calculate_shear_pos_select(self, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        n_i = cat_i.nobj

        cat_j = self.get_shear_catalog(j)
        cat_j = self.touch_patches(cat_j)
        rancat_j = self.get_random_catalog(j)
        rancat_j = self.touch_patches(rancat_j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        if self.rank == 0:
            print(f"Calculating shear-position selected bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")

        if n_i == 0 or n_j == 0:
            if self.rank == 0:
                print("Empty catalog: returning None")
                return None

        ng = treecorr.NGCorrelation(self.config, max_rpar=0.0)
        t1 = perf_counter()
        ng.process(cat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])

        if rancat_j:
            rg = treecorr.NGCorrelation(self.config, max_rpar=0.0)
            rg.process(rancat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])
        else:
            rg = None

        ng.calculateXi(rg=rg)
        t2 = perf_counter()
        if self.rank == 0:
            print(f"Processing took {t2 - t1:.1f} seconds")

        return ng

    def calculate_pos_pos(self, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        rancat_i = self.get_random_catalog(i)
        rancat_i = self.touch_patches(rancat_i)
        n_i = cat_i.nobj
        n_rand_i = rancat_i.nobj if rancat_i is not None else 0

        if i == j:
            cat_j = None
            rancat_j = rancat_i
            n_j = n_i
            n_rand_j = n_rand_i
        else:
            cat_j = self.get_shear_catalog(j)
            cat_j = self.touch_patches(cat_j)
            rancat_j = self.get_random_catalog(j)
            rancat_j = self.touch_patches(rancat_j)
            n_j = cat_j.nobj
            n_rand_j = rancat_j.nobj

        if self.config['use_subsampled_randoms']:
            rancat_sub_i = self.get_subsampled_random_catalog(i)
            rancat_sub_i = self.touch_patches(rancat_sub_i)
            n_rand_sub_i = rancat_sub_i.nobj if rancat_sub_i is not None else 0

            if i == j:
                rancat_sub_j = rancat_sub_i
                n_rand_sub_j = n_rand_sub_i
            else:
                rancat_sub_j = self.get_subsampled_random_catalog(j)
                rancat_sub_j = self.touch_patches(rancat_sub_j)
                n_rand_sub_j = rancat_sub_j.nobj if rancat_sub_j is not None else 0

        if self.rank == 0:
            print(
                f"Calculating source-source bin pair ({i}, {j}): {n_i} x {n_j} objects,  {n_rand_i} x {n_rand_j} randoms"
            )
            if self.config["use_subsampled_randoms"]:
                print(f"and for the rr term, {n_rand_sub_i} x {n_rand_sub_j} pairs")

        if n_i == 0 or n_j == 0:
            if self.rank == 0:
                print("Empty catalog: returning None")
            return None
        
        t1 = perf_counter()

        nn = treecorr.NNCorrelation(self.config)
        nn.process(cat_i, cat_j, comm=self.comm, low_mem=self.config["low_mem"])

        nr = treecorr.NNCorrelation(self.config)
        nr.process(cat_i, rancat_j, comm=self.comm, low_mem=self.config["low_mem"])

        if i == j:
            rancat_j = None
            rancat_sub_j = None

        rr = treecorr.NNCorrelation(self.config)
        if self.confif["use_subsampled_randoms"]:
            rr.process(rancat_sub_i, rancat_sub_j, comm=self.comm, low_mem=self.config["low_mem"])
        else:
            rr.process(rancat_i, rancat_j, comm=self.comm, low_mem=self.config["low_mem"])

        if i == j:
            rn = None
        else:
            rn = treecorr.NNCorrelation(self.config)
            rn.process(rancat_i, cat_j, comm=self.comm, low_mem=self.config["low_mem"])

        t2 = perf_counter()
        nn.calculateXi(rr, dr=nr, rd=rn)
        if self.rank == 0:
            print(f"Processing took {t2-t1:.1f} seconds")

        return nn