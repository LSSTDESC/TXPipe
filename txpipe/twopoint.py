from .base_stage import PipelineStage
from .data_types import (
    HDFFile,
    ShearCatalog,
    SACCFile,
    TextFile,
    MapsFile,
    QPNOfZFile,
)
from .utils.patches import PatchMaker
import numpy as np
import collections
import sys
import os
import pathlib
from time import perf_counter
import gc
from .utils import choose_pixelization

# This creates a little mini-type, like a struct,
# for holding individual measurements
Measurement = collections.namedtuple("Measurement", ["corr_type", "object", "i", "j"])

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

#external cross correlations
POS_EXT = 3
SHEAR_EXT = 4

class TXTwoPoint(PipelineStage):
    """
    Make 2pt measurements using TreeCorr

    This stage make the full set of cosmic shear, galaxy-galaxy lensing, 
    and galaxy density measurements on the tomographic catalog using TreeCorr.

    Results are saved to a sacc file.
    """
    name = "TXTwoPoint"
    inputs = [
        ("binned_shear_catalog", ShearCatalog),
        ("binned_lens_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("binned_random_catalog_sub", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [("twopoint_data_real_raw", SACCFile), ("twopoint_gamma_x", SACCFile)]
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
        "auto_only": False,
        "var_method": "jackknife",
        "use_randoms": True,
        "low_mem": False,
        "patch_dir": "./cache/patches",
        "chunk_rows": 100_000,
        "share_patch_files": False,
        "metric": "Euclidean",
        "gaussian_sims_factor": [1.], 
        "use_subsampled_randoms": True, #use subsampled randoms file for RR
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        import sacc
        import healpy
        import treecorr

        # Binning information
        source_list, lens_list = self.read_nbin()

        if self.rank == 0:
            # This is a workaround for the fact the the ceci config stuff doesn't
            # quite handle the get method properly.
            # Which metrics are available, and how they are interpreted, depends on
            # whether a distance is in the catalogs returned in get_shear_catalog
            # and friends, below. In this base class only the 2D metrics will be
            # available, but subclasses can specify to load a distance column too.
            metric = self.config["metric"] if "metric" in self.config else "Euclidean"
            print(f"Running TreeCorr with metric \"{metric}\"")

        # Calculate metadata like the area and related
        # quantities
        meta = self.read_metadata()

        # Choose which pairs of bins to calculate
        calcs = self.select_calculations(source_list, lens_list)
        sys.stdout.flush()

        # Split the catalogs into patch files
        self.prepare_patches(calcs, meta)

        results = []
        for i, j, k in calcs:
            result = self.call_treecorr(i, j, k)
            results.append(result)

        if self.comm:
            self.comm.Barrier()

        # Save the results
        self.write_output(source_list, lens_list, meta, results)

    def select_calculations(self, source_list, lens_list):
        calcs = []

        # For shear-shear we omit pairs with j>i
        if self.config["do_shear_shear"]:
            print('DOING SHEAR-SHEAR')
            k = SHEAR_SHEAR
            for i in source_list:
                for j in range(i + 1):
                    if j in source_list:
                        calcs.append((i, j, k))

        # For shear-position we use all pairs
        if self.config["do_shear_pos"]:
            print('DOING SHEAR-POS')
            k = SHEAR_POS
            for i in source_list:
                for j in lens_list:
                    calcs.append((i, j, k))

        # For position-position we omit pairs with j>i
        if self.config["do_pos_pos"]:
            print('DOING POS-POS')
            if not self.config["use_randoms"]:
                raise ValueError(
                    "You need to have a random catalog to calculate position-position correlations"
                )
            k = POS_POS
            if self.config["auto_only"]:
                for i in lens_list:
                    calcs.append((i, i, k))
            else:
                for i in lens_list:
                    for j in range(i + 1):
                        if j in lens_list:
                            calcs.append((i, j, k))

        if self.rank == 0:
            print(f"Running {len(calcs)} calculations: {calcs}")

        return calcs

    def read_nbin(self):
        """
        Determine the bins to use in this analysis, either from the input file
        or from the configuration.
        """
        if self.config["source_bins"] == [-1] and self.config["lens_bins"] == [-1]:
            source_list, lens_list = self._read_nbin_from_tomography()
        else:
            source_list, lens_list = self._read_nbin_from_config()

        ns = len(source_list)
        nl = len(lens_list)
        if self.rank == 0:
            print(f"Running with {ns} source bins and {nl} lens bins")

        return source_list, lens_list

    # These two functions can be combined into a single one.
    def _read_nbin_from_tomography(self):
        if self.get_input("binned_shear_catalog") == "none":
            nbin_source = 0
        else:
            with self.open_input("binned_shear_catalog") as f:
                nbin_source = f["shear"].attrs["nbin_source"]

        if self.get_input("binned_lens_catalog") == "none":
            nbin_lens = 0
        else:
            with self.open_input("binned_lens_catalog") as f:
                nbin_lens = f["lens"].attrs["nbin_lens"]

        source_list = list(range(nbin_source))
        lens_list = list(range(nbin_lens))

        return source_list, lens_list

    def _read_nbin_from_config(self):
        # TODO handle the case where the user only specefies
        # bins for only sources or only lenses
        source_list = self.config["source_bins"]
        lens_list = self.config["lens_bins"]

        # catch bad input
        tomo_source_list, tomo_lens_list = self._read_nbin_from_tomography()
        tomo_nbin_source = len(tomo_source_list)
        tomo_nbin_lens = len(tomo_lens_list)

        nbin_source = len(source_list)
        nbin_lens = len(lens_list)

        if source_list == [-1]:
            source_list = tomo_source_list
        if lens_list == [-1]:
            lens_list = tomo_lens_list

        # if more bins are input than exist, raise an error
        if not nbin_source <= tomo_nbin_source:
            raise ValueError(
                f"Requested too many source bins in the config ({nbin_source}): max is {tomo_nbin_source}"
            )
        if not nbin_lens <= tomo_nbin_lens:
            raise ValueError(
                f"Requested too many lens bins in the config ({nbin_lens}): max is {tomo_nbin_lens}"
            )

        # make sure the bin numbers actually exist
        for i in source_list:
            if i not in tomo_source_list:
                raise ValueError(
                    f"Requested source bin {i} that is not in the input file"
                )

        for i in lens_list:
            if i not in tomo_lens_list:
                raise ValueError(
                    f"Requested lens bin {i} that is not in the input file"
                )

        return source_list, lens_list


    def add_data_points(self, S, results):
        import treecorr
        import sacc

        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        WTHETA = sacc.standard_types.galaxy_density_xi

        comb = []
        for index, d in enumerate(results):
            # First the tracers and generic tags
            tracer1 = f"source_{d.i}" if d.corr_type in [XI, GAMMAT] else f"lens_{d.i}"
            tracer2 = f"source_{d.j}" if d.corr_type in [XI] else f"lens_{d.j}"

            # This happens when there is an empty bin. We can't do a covariance
            # here, or anything useful, really, so we just skip this bin.
            if d.object is None:
                continue

            theta = np.exp(d.object.meanlogr)
            npair = d.object.npairs
            weight = d.object.weight
            # xip / xim is a special case because it has two observables.
            # the other two are together below
            if d.corr_type == XI:
                xip = d.object.xip
                xim = d.object.xim
                xiperr = np.sqrt(d.object.varxip)
                ximerr = np.sqrt(d.object.varxim)
                n = len(xip)
                # add all the data points to the sacc
                for i in range(n):
                    S.add_data_point(
                        XIP,
                        (tracer1, tracer2),
                        xip[i],
                        theta=theta[i],
                        error=xiperr[i],
                        npair=npair[i],
                        weight=weight[i],
                    )
                for i in range(n):
                    S.add_data_point(
                        XIM,
                        (tracer1, tracer2),
                        xim[i],
                        theta=theta[i],
                        error=ximerr[i],
                        npair=npair[i],
                        weight=weight[i],
                    )
            else:
                if self.config['gaussian_sims_factor'] != [1.]:
                    # only for gammat and wtheta, for the gaussian simulations we need to scale the measurements up to correct for
                    # the scaling of the density field when building the simulations.
                    if 'lens' in tracer2:
                        if 'lens' in tracer1:
                            scaling_factor = self.config['gaussian_sims_factor'][int(tracer1[-1])]*self.config['gaussian_sims_factor'][int(tracer2[-1])]
                        else:
                            scaling_factor = self.config['gaussian_sims_factor'][int(tracer2[-1])]
                            
                    d.object.xi *=scaling_factor
                    d.object.varxi *=(scaling_factor**2)
                    
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
                        weight=weight[i],
                    )
                    
            # We build up the comb list to get the covariance of it later
            # in the same order as our data points
            comb.append(d.object)


                    
        # Add the covariance.  There are several different jackknife approaches
        # available - see the treecorr docs
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

        XI = "combined"
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x

        covs = []
        for index, d in enumerate(results):
            tracer1 = (
                f"source_{d.i}" if d.corr_type in [XI, GAMMAT] else f"lens_{d.i}"
            )
            tracer2 = f"source_{d.j}" if d.corr_type in [XI] else f"lens_{d.j}"

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

        S.add_covariance(covs)


    def write_output(self, source_list, lens_list, meta, results):
        import sacc
        import treecorr

        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        WTHETA = sacc.standard_types.galaxy_density_xi

        S = sacc.Sacc()
        S2 = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data

        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        if self.config["do_shear_pos"] or self.config["do_shear_shear"]:
            if source_list:
                with self.open_input("shear_photoz_stack", wrapper=True) as f:
                    for i in source_list:
                        z, Nz = f.get_bin_n_of_z(i)
                        S.add_tracer("NZ", f"source_{i}", z, Nz)
                        if self.config["do_shear_pos"] == True:
                            S2.add_tracer("NZ", f"source_{i}", z, Nz)

        if self.config["do_pos_pos"] or self.config["do_shear_pos"]:
            if lens_list:
                with self.open_input("lens_photoz_stack", wrapper=True) as f:
                    # For both source and lens
                    for i in lens_list:
                        z, Nz = f.get_bin_n_of_z(i)
                        S.add_tracer("NZ", f"lens_{i}", z, Nz)
                        if self.config["do_shear_pos"] == True:
                            S2.add_tracer("NZ", f"lens_{i}", z, Nz)

        # Now build up the collection of data points, adding them all to
        # the sacc data one by one.
        self.add_data_points(S, results)

        # The other processes are only needed for the covariance estimation.
        # They do a bunch of other stuff here that isn't actually needed, but
        # it should all be very fast. After this point they are not needed
        # at all so return
        if self.rank != 0:
            return


        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()


        self.write_metadata(S, meta)


        # Finally, save the output to Sacc file
        S.save_fits(self.get_output("twopoint_data_real_raw"), overwrite=True)


        # Adding the gammaX calculation:
        if self.config["do_shear_pos"] == True:
            self.add_gamma_x_data_points(S2, results)
            S2.to_canonical_order()
            self.write_metadata(S2, meta)
        # always write the file, even if it is empty
        S2.save_fits(self.get_output("twopoint_gamma_x"), overwrite=True)

    def write_metadata(self, S, meta):
        # We also save the associated metadata to the file
        for k, v in meta.items():
            if np.isscalar(v):
                S.metadata[k] = v
            else:
                for i, vi in enumerate(v):
                    S.metadata[f"{k}_{i}"] = vi

        # Add provenance metadata.  In managed formats this is done
        # automatically, but because the Sacc library is external
        # we do it manually here.
        provenance = self.gather_provenance()
        provenance.update(SACCFile.generate_provenance())
        for key, value in provenance.items():
            if isinstance(value, str) and "\n" in value:
                values = value.split("\n")
                for i, v in enumerate(values):
                    S.metadata[f"provenance/{key}_{i}"] = v
            else:
                S.metadata[f"provenance/{key}"] = value

    def call_treecorr(self, i, j, k):
        """
        This is a wrapper for interaction with treecorr.
        """
        import sacc
        import pickle
        #TODO: fix up the caching code
        if self.name == "TXTwoPoint" or self.name == "TXTwoPointPixel":
            pickle_filename = self.get_output("twopoint_data_real_raw") + f".checkpoint-{i}-{j}-{k}.pkl"
            #pickle_filename = f"treecorr-cache-{i}-{j}-{k}.pkl"

            if os.path.exists(pickle_filename):
                print(f"{self.rank} WARNING USING THIS PICKLE FILE I FOUND: {pickle_filename}")
                with open(pickle_filename, "rb") as f:
                    result = pickle.load(f)
                return result
 
        if k == SHEAR_SHEAR:
            xx = self.calculate_shear_shear(i, j)
            xtype = "combined"
        elif k == SHEAR_POS:
            xx = self.calculate_shear_pos(i, j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k == POS_POS:
            xx = self.calculate_pos_pos(i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        else:
            raise ValueError(f"Unknown correlation function {k}")

        # Force garbage collection here to make sure all the
        # catalogs are definitely freed
        gc.collect()

        # The measurement object collects the results and type info.
        # we use it because the ordering will not be simple if we have
        # parallelized, so it's good to keep explicit track.
        result = Measurement(xtype, xx, i, j)

        sys.stdout.flush()

        if self.comm:
            self.comm.Barrier()

        if self.name == "TXTwoPoint" or self.name == "TXTwoPointPixel":
            if self.rank == 0:
                print(f"Pickling result to {pickle_filename}")
                with open(pickle_filename, "wb") as f:
                    pickle.dump(result, f)
 
        return result

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
        # Make the full list of catalogs to run
        cats = set()

        # Use shear-shear and pos-pos only here as they represent
        # catalogs not pairs.
        for i, j, k in calcs:
            if k == SHEAR_SHEAR:
                cats.add((i, SHEAR_SHEAR))
                cats.add((j, SHEAR_SHEAR))
            elif k == SHEAR_POS:
                cats.add((i, SHEAR_SHEAR))
                cats.add((j, POS_POS))
            elif k == POS_POS:
                cats.add((i, POS_POS))
                cats.add((j, POS_POS))
        cats = list(cats)
        cats.sort(key=str)

        chunk_rows = self.config["chunk_rows"]
        npatch_shear = 0
        npatch_pos = 0
        npatch_ran = 0

        self.empty_patch_exists = {}

        # Parallelization is now done at the patch level
        for (h, k) in cats:
            ktxt = "shear" if k == SHEAR_SHEAR else "position"
            print(f"Rank {self.rank} making patches for {ktxt} catalog bin {h}")

            # For shear we just have the one catalog. For position we may
            # have randoms also. We explicitly delete catalogs after loading
            # them to ensure we don't have two in memory at once.
            if k == SHEAR_SHEAR:
                cat = self.get_shear_catalog(h)
                npatch_shear,contains_empty = PatchMaker.run(cat, chunk_rows, self.comm)
                self.empty_patch_exists[cat.save_patch_dir] = contains_empty
                del cat
            else:
                cat = self.get_lens_catalog(h)
                npatch_pos,contains_empty = PatchMaker.run(cat, chunk_rows, self.comm)
                self.empty_patch_exists[cat.save_patch_dir] = contains_empty
                del cat

                ran_cat = self.get_random_catalog(h)
                # support use_randoms = False
                if ran_cat is None:
                    continue
                npatch_ran,contains_empty = PatchMaker.run(ran_cat, chunk_rows, self.comm)
                self.empty_patch_exists[ran_cat.save_patch_dir] = contains_empty
                del ran_cat

                if self.config["use_subsampled_randoms"]:
                    ran_cat = self.get_subsampled_random_catalog(h)
                    npatch_ran,contains_empty = PatchMaker.run(ran_cat, chunk_rows, self.comm)
                    self.empty_patch_exists[ran_cat.save_patch_dir] = contains_empty
                    del ran_cat

        meta["npatch_shear"] = npatch_shear
        meta["npatch_pos"] = npatch_pos
        meta["npatch_ran"] = npatch_ran
        # stop other processes progressing to the rest of the code and
        # trying to load things we have not written yet
        if self.comm is not None:
            self.comm.Barrier()

    def get_patch_dir(self, input_tag, b):
        """
        Select a patch directory for the file  with the given input tag
        and with a bin number/label.

        To ensure that if you change the catalog the patch dir will also
        change, the directory path includes the unique ID of the input file.

        Parameters
        ----------
        input_tag: str
            One of the tags in the class's inputs attribute
        b: any
            An additional label used as the last component in the returned
            directory

        Returns
        -------
        str: a directory, which has been created if it did not exist already.
        """
        # start from a user-specified base directory
        patch_base = self.config["patch_dir"]

        # append the unique identifier for the parent catalog file
        with self.open_input(input_tag, wrapper=True) as f:
            p = f.read_provenance()
            uuid = p["uuid"]
            pth = pathlib.Path(f.path).resolve()
            ctime = os.stat(pth).st_ctime

        # We expect the input files to be generated within a pipeline and so always
        # have input files to have a unique ID.  But if for some reason it doesn't
        # have one we handle that too.
        if uuid == "UNKNOWN":
            ident = hash(f"{pth}{ctime}").to_bytes(8, "big", signed=True).hex()
            name = f"{input_tag}_{ident}"
        else:
            name = f"{input_tag}_{uuid}"

        # Include a tag for the current stage name, so that
        # if we are running several subclasses at the same time
        # they don't interfere with each other. This is a waste of
        # disc space, but hopefully we are not short of that.
        if not self.config["share_patch_files"]:
            name = self.instance_name + name

        # And finally append the bin name or number
        patch_dir = pathlib.Path(patch_base) / name / str(b)

        # Make the directory and return it
        pathlib.Path(patch_dir).mkdir(exist_ok=True, parents=True)
        return patch_dir

    def get_shear_catalog(self, i):
        import treecorr

        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            self.get_input("binned_shear_catalog"),
            ext=f"/shear/bin_{i}",
            g1_col="g1",
            g2_col="g2",
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


    def get_lens_catalog(self, i):
        import treecorr

        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            self.get_input("binned_lens_catalog"),
            ext=f"/lens/bin_{i}",
            ra_col="ra",
            dec_col="dec",
            w_col="weight",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_lens_catalog", i),
        )

        return cat

    def get_random_catalog(self, i):
        import treecorr

        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("binned_random_catalog"),
            ext=f"/randoms/bin_{i}",
            ra_col="ra",
            dec_col="dec",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_random_catalog", i),
        )

        return rancat

    def get_subsampled_random_catalog(self, i):
        import treecorr

        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("binned_random_catalog_sub"),
            ext=f"/randoms/bin_{i}",
            ra_col="ra",
            dec_col="dec",
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_random_catalog_sub", i),
        )

        return rancat

    def touch_patches(self, cat):
        # If any patches were empty for this cat
        # run get_patches on rank 0 and bcast
        # this will re-make patches but prevents processes conflicting 
        # in the gg.process
        # If no patches are empty returns the cat, unaltered
        if cat is None:
            return cat

        if self.empty_patch_exists[cat.save_patch_dir]:
            if self.rank==0:
                cat.get_patches()
            if self.comm is not None:
                cat = self.comm.bcast(cat, root=0)

        return cat

    def calculate_shear_shear(self, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        n_i = cat_i.nobj

        if i == j:
            cat_j = None
            n_j = n_i
        else:
            cat_j = self.get_shear_catalog(j)
            cat_j = self.touch_patches(cat_j)
            n_j = cat_j.nobj


        if self.rank == 0:
            print(
                f"Calculating shear-shear bin pair ({i},{j}): {n_i} x {n_j} objects using MPI"
            )

        if n_i == 0 or n_j == 0:
            if self.rank == 0:
                print("Empty catalog: returning None")
            return None

        gg = treecorr.GGCorrelation(self.config)
        t1 = perf_counter()
        gg.process(cat_i, cat_j, low_mem=self.config["low_mem"], comm=self.comm)
        t2 = perf_counter()
        if self.rank == 0:
            print(f"Processing took {t2 - t1:.1f} seconds")

        return gg

    def calculate_shear_pos(self, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        cat_i = self.touch_patches(cat_i)
        n_i = cat_i.nobj

        cat_j = self.get_lens_catalog(j)
        cat_j = self.touch_patches(cat_j)
        rancat_j = self.get_random_catalog(j)
        rancat_j = self.touch_patches(rancat_j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        if self.rank == 0:
            print(
                f"Calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms"
            )

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

    def calculate_pos_pos(self, i, j):
        import treecorr

        cat_i = self.get_lens_catalog(i)
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
            cat_j = self.get_lens_catalog(j)
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
                f"Calculating position-position bin pair ({i}, {j}): {n_i} x {n_j} objects,  {n_rand_i} x {n_rand_j} randoms"
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

        # The next calculation is faster if we explicitly tell TreeCorr
        # that its two catalogs here are the same one.
        if i == j:
            rancat_j = None
            n_rand_sub_j = None

        rr = treecorr.NNCorrelation(self.config)
        if self.config["use_subsampled_randoms"]:
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
            print(f"Processing took {t2 - t1:.1f} seconds")

        return nn

    def read_metadata(self):
        meta_data = self.open_input("tracer_metadata")
        area = meta_data["tracers"].attrs["area"]
        meta = {}
        meta["area"] = area
        try:
            sigma_e = meta_data["tracers/sigma_e"][:]
            N_eff = meta_data["tracers/N_eff"][:]
            mean_e1 = meta_data["tracers/mean_e1"][:]
            mean_e2 = meta_data["tracers/mean_e2"][:]

            meta["neff"] = N_eff
            meta["area"] = area
            meta["sigma_e"] = sigma_e
            meta["mean_e1"] = mean_e1
            meta["mean_e2"] = mean_e2
        
        except KeyError: #will happen for lens only runs
            pass
        
        return meta



class TXTwoPointPixel(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses maps to compute
    pixelized versions of the real space correlation functions.
    This is useful when the number density of the galaxy samples
    is too high to use random points to sample the mask.
    """

    name = "TXTwoPointPixel"
    inputs = [
        ("density_maps", MapsFile),
        ("source_maps", MapsFile),
        ("binned_shear_catalog", ShearCatalog),
        ("binned_lens_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
        ("mask", MapsFile),
    ]
    outputs = [("twopoint_data_real_raw", SACCFile), ("twopoint_gamma_x", SACCFile)]
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
        "use_subsampled_randoms":False, #not used for pixel estimator
    }
    

    def get_density_map(self, i):
        import treecorr
        
        with self.open_input("density_maps", wrapper=True) as f:
            info = f.read_map_info(f"delta_{i}")
            map_d, pix, nside = f.read_healpix(f"delta_{i}", return_all=True)
            print(f"Loaded {i} overdensity maps")

        # Read the mask to get fracdet weights
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")

        scheme = choose_pixelization(**info) 
        ra_pix, dec_pix = scheme.pix2ang(pix)

        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            ra=ra_pix,
            dec=dec_pix,
            w=mask[pix], #weight pixels by their fracdet
            k=map_d[pix],
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

class TXTwoPointPixelExtCross(TXTwoPointPixel):
    """
    TXTwoPointPixel - External - Cross correlation
    This subclass of TXTwoPointPixel uses maps to compute
    cross correlations between the galaxy density maps and 
    an external set of Survey Property (or other contaminant) 
    maps
    """

    name = "TXTwoPointPixelExtCross"
    inputs = [
        ("density_maps", MapsFile),
        ("source_maps", MapsFile),
        ("binned_shear_catalog", ShearCatalog),
        ("binned_lens_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("patch_centers", TextFile),
        ("tracer_metadata", HDFFile),
        ("mask", MapsFile),
    ]
    outputs = [("twopoint_data_ext_cross_raw", SACCFile)]
    # Add values to the config file that are not previously defined
    config_options = {
        "supreme_path_root": "",
        "do_pos_ext": True,
        "do_shear_ext": True,

        # TODO: Remove any unnesesary config options here
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
        "do_shear_shear": False,
        "do_shear_pos": False,
        "do_pos_pos": False,
        "var_method": "jackknife",
        "low_mem": False,
        "patch_dir": "./cache/patches",
        "chunk_rows": 100_000,
        "share_patch_files": False,
        "metric": "Euclidean",
        "use_randoms": True,
        "auto_only": False,
        "gaussian_sims_factor": [1.], 
        "use_subsampled_randoms":False, #not used for pixel estimator
    }

    def get_ext_list(self):
        import glob
        root = self.config["supreme_path_root"]
        sys_files = glob.glob(f"{root}*.hs")
        return sys_files

    def select_calculations(self, source_list, lens_list):
        """
        For TXTwoPointPixelExtCross, this method only selects
        the cross correlations between data and external maps
        """
        calcs = []

        #get the list of external map files
        self.ext_list = self.get_ext_list()
        ext_list = np.arange(len( self.ext_list ))

        # For shear-ext
        if self.config["do_shear_ext"]:
            k = SHEAR_EXT
            for i in source_list:
                for j in ext_list:
                    calcs.append((i, j, k))

        if self.config["do_pos_ext"]:
            k = POS_EXT
            for i in lens_list:
                for j in ext_list:
                    calcs.append((i, j, k))

        if self.rank == 0:
            print(f"Running {len(calcs)} calculations: {calcs}")

        return calcs

    def call_treecorr(self, i, j, k):
        """
        call_treecorr is modified for this sub-class to include the external-cross-correlations

        This is a wrapper for interaction with treecorr.
        """
        import sacc
        import pickle
        #TODO: fix up the caching code
        if self.name == "TXTwoPointPixelExtCross": 
            pickle_filename = self.get_output("twopoint_data_ext_cross_raw") + f".checkpoint-{i}-{j}-{k}.pkl"

            if os.path.exists(pickle_filename):
                print(f"{self.rank} WARNING USING THIS PICKLE FILE I FOUND: {pickle_filename}")
                with open(pickle_filename, "rb") as f:
                    result = pickle.load(f)
                return result
 
        if k == SHEAR_SHEAR:
            xx = self.calculate_shear_shear(i, j)
            xtype = "combined"
        elif k == SHEAR_POS:
            xx = self.calculate_shear_pos(i, j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k == POS_POS:
            xx = self.calculate_pos_pos(i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        elif k == POS_EXT:
            assert self.name == "TXTwoPointPixelExtCross"
            xx = self.calculate_pos_ext(i, j)
            xtype = sacc.build_data_type_name( "galaxy", ["density", "ext"], "xi" )
        elif k == SHEAR_EXT:
            assert self.name == "TXTwoPointPixelExtCross"
            xx = self.calculate_shear_ext(i, j)
            xtype = sacc.build_data_type_name( "galaxy", ["shear", "ext"], "xi" )
        else:
            raise ValueError(f"Unknown correlation function {k}")

        # Force garbage collection here to make sure all the
        # catalogs are definitely freed
        gc.collect()

        # The measurement object collects the results and type info.
        # we use it because the ordering will not be simple if we have
        # parallelized, so it's good to keep explicit track.
        result = Measurement(xtype, xx, i, j)

        sys.stdout.flush()

        if self.comm:
            self.comm.Barrier()

        if self.name == "TXTwoPointPixelExtCross":
            if self.rank == 0:
                print(f"Pickling result to {pickle_filename}")
                with open(pickle_filename, "wb") as f:
                    pickle.dump(result, f)
 
        return result

    def calculate_pos_ext(self, i, j):
        import treecorr

        cat_i = self.get_density_map(i)

        cat_j = self.get_ext_map(j)

        if self.rank == 0:
            print(
                f"Calculating position-external bin pair ({i}, {j})"
            )

        t1 = perf_counter()

        kk = treecorr.KKCorrelation(self.config)
        kk.process(cat_i, cat_j, comm=self.comm, low_mem=self.config["low_mem"])

        return kk


    def calculate_shear_ext(self, i, j):
        import treecorr

        cat_i = self.get_shear_map(i)

        cat_j = self.get_ext_map(j)

        if self.rank == 0:
            print(
                f"Calculating shear-external bin pair ({i},{j})."
            )

        kg = treecorr.KGCorrelation(self.config)
        t1 = perf_counter()
        kg.process(cat_j, cat_i, comm=self.comm, low_mem=self.config["low_mem"])

        return kg

    def get_ext_map(self, i):
        """
        get the ith external map (e.g. SP map) from the directory specified in the config 
        """
        import treecorr
        import healsparse
        import healpy as hp 
        
        # Read the mask to get fracdet weights
        with self.open_input("mask", wrapper=True) as f:
            info = f.read_map_info('mask')
            mask, pix, nside = f.read_healpix('mask', return_all=True)

        # open the input healsparse map
        sys_files = self.get_ext_list()
        sys_file = sys_files[i]
        m = healsparse.HealSparseMap.read(sys_file)
        m = m.degrade(nside)
        if info['nest'] == True:
            map_d = m[pix]
        else:
            map_d = m[hp.ring2nest(nside, pix)] 

        #convert SP map to an overdensity
        mean_data = np.average(map_d, weights=mask[pix] )
        map_d = map_d/mean_data - 1.

        scheme = choose_pixelization(**info) 
        ra_pix, dec_pix = scheme.pix2ang(pix)

        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            ra=ra_pix,
            dec=dec_pix,
            w=mask[pix], #weight pixels by their fracdet
            k=map_d,
            ra_units="degree",
            dec_units="degree",
            patch_centers=self.get_input("patch_centers"),
        )
        return cat

    def add_data_points(self, S, results, tracers_later=False):
        """
        modify add_data_points to know about external map cross correlations
        and allow tracers_later=True
        """
        import treecorr
        import sacc

        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        WTHETA = sacc.standard_types.galaxy_density_xi
        POS_EXT_TYPE = sacc.build_data_type_name( "galaxy", ["density", "ext"], "xi" )
        SHEAR_EXT_TYPE = sacc.build_data_type_name( "galaxy", ["shear", "ext"], "xi" )

        comb = []
        for index, d in enumerate(results):
            # First the tracers and generic tags
            if d.corr_type in [XI,XIP,XIM]:
                tracer1 = f"source_{d.i}"
                tracer2 = f"source_{d.j}"
            elif d.corr_type in [GAMMAT, GAMMAX]:
                tracer1 = f"source_{d.i}"
                tracer2 = f"lens_{d.j}"
            elif d.corr_type == WTHETA:
                tracer1 = f"lens_{d.i}"
                tracer2 = f"lens_{d.j}"
            elif d.corr_type == POS_EXT_TYPE:
                tracer1 = f"lens_{d.i}"
                tracer2 = f"external_{d.j}"
            elif d.corr_type == SHEAR_EXT_TYPE:
                tracer1 = f"source_{d.i}"
                tracer2 = f"external_{d.j}"
            else:
                raise RuntimeError('unrecognised corr_type')

            # This happens when there is an empty bin. We can't do a covariance
            # here, or anything useful, really, so we just skip this bin.
            if d.object is None:
                continue

            theta = np.exp(d.object.meanlogr)
            npair = d.object.npairs
            weight = d.object.weight
            # xip / xim is a special case because it has two observables.
            # the other two are together below
            if d.corr_type == XI:
                xip = d.object.xip
                xim = d.object.xim
                xiperr = np.sqrt(d.object.varxip)
                ximerr = np.sqrt(d.object.varxim)
                n = len(xip)
                # add all the data points to the sacc
                for i in range(n):
                    S.add_data_point(
                        XIP,
                        (tracer1, tracer2),
                        xip[i],
                        theta=theta[i],
                        error=xiperr[i],
                        npair=npair[i],
                        weight=weight[i],
                        tracers_later=tracers_later,
                    )
                for i in range(n):
                    S.add_data_point(
                        XIM,
                        (tracer1, tracer2),
                        xim[i],
                        theta=theta[i],
                        error=ximerr[i],
                        npair=npair[i],
                        weight=weight[i],
                        tracers_later=tracers_later,
                    )
            else:
                if self.config['gaussian_sims_factor'] != [1.]:
                    # only for gammat and wtheta, for the gaussian simulations we need to scale the measurements up to correct for
                    # the scaling of the density field when building the simulations.
                    if 'lens' in tracer2:
                        if 'lens' in tracer1:
                            scaling_factor = self.config['gaussian_sims_factor'][int(tracer1[-1])]*self.config['gaussian_sims_factor'][int(tracer2[-1])]
                        else:
                            scaling_factor = self.config['gaussian_sims_factor'][int(tracer2[-1])]
                            
                    d.object.xi *=scaling_factor
                    d.object.varxi *=(scaling_factor**2)
                    
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
                        weight=weight[i],
                        tracers_later=tracers_later,
                    )
                    
            # We build up the comb list to get the covariance of it later
            # in the same order as our data points
            comb.append(d.object)


                    
        # Add the covariance.  There are several different jackknife approaches
        # available - see the treecorr docs
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


    def write_output(self, source_list, lens_list, meta, results):
        """
        Re-define method to use external cross-correlation outputs 
        """
        import sacc
        import treecorr

        S = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data

        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        if source_list:
            with self.open_input("shear_photoz_stack", wrapper=True) as f:
                for i in source_list:
                    z, Nz = f.get_bin_n_of_z(i)
                    S.add_tracer("NZ", f"source_{i}", z, Nz)

        if lens_list:
            with self.open_input("lens_photoz_stack", wrapper=True) as f:
                # For both source and lens
                for i in lens_list:
                    z, Nz = f.get_bin_n_of_z(i)
                    S.add_tracer("NZ", f"lens_{i}", z, Nz)

        # Now build up the collection of data points, adding them all to
        # the sacc data one by one.
        self.add_data_points(S, results, tracers_later=True )

        # The other processes are only needed for the covariance estimation.
        # They do a bunch of other stuff here that isn't actually needed, but
        # it should all be very fast. After this point they are not needed
        # at all so return
        if self.rank != 0:
            return


        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()


        #add list of sp map names to the metadata
        meta['ext_list'] = self.ext_list

        self.write_metadata(S, meta)


        # Finally, save the output to Sacc file
        S.save_fits(self.get_output("twopoint_data_ext_cross_raw"), overwrite=True)




if __name__ == "__main__":
    PipelineStage.main()
