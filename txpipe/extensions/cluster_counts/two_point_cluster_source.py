from ...twopoint import TXTwoPoint
from ...data_types import (
    HDFFile,
    ShearCatalog,
    TomographyCatalog,
    PNGFile,
    TextFile,
    PickleFile,
)
import numpy as np

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

class TXTwoPointClusterSource(TXTwoPoint):
    """
    Measure 2-pt shear-position using the Rlens metric

    The Rlens metric uses the impact factor between the vector to the source galaxy
    and the location of the lens galaxy as its distance.

    Compared to the parent TXTwoPoint class this:
    - does only the shear-position correlation
    - loads comoving coordinates as the radial distance in the lens and random catalogs
    - removes the sep_unit configuration option since the unit must always be Mpc
    - changes the default metric
    - does not yet save any results correctly, just prints them out.

    This is mainly an example stage for future CL work.

    """
    name = "TXTwoPointClusterSource"
    inputs = [
        ("binned_lens_catalog", HDFFile),
        ("binned_shear_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("shear_tomography_catalog", TomographyCatalog),
    ]
    #    ("patch_centers", TextFile),
    #]
    outputs = [
        ("cluster_profiles",  PickleFile),       
    ]

    config_options = {
        # TODO: Allow more fine-grained selection of 2pt subsets to compute
        "calcs": [0, 1, 2],
        "min_sep": 0.1,  # arcmin
        "max_sep": 200.0,  # arcmin
        "nbins": 9,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "flip_g1": False,
        "flip_g2": True,
        "cores_per_task": 20,
        "verbose": 1,
        "source_bins": [-1],
        "lens_bins": [-1],
        "reduce_randoms_size": 1.0,
        "do_shear_shear": False,
        "do_shear_pos": True,
        "do_pos_pos": False,
        "var_method": "jackknife",
        "use_randoms": True,
        "low_mem": False,
        "patch_dir": "./cache/patches",
        "chunk_rows": 100_000,
        "share_patch_files": False,
        "metric": "Euclidean",
        "use_subsampled_randoms": False,
    }

    def read_metadata(self):
        return {}


    def get_lens_catalog(self, i):
        # Override the lens catalog generation.
        # This is like the parent version except we also add the r_col keyword
        import treecorr
        bin_name = None
        with self.open_input("binned_lens_catalog") as f:
            for j, zbin_richbin in enumerate(f['cluster_bin'].keys()):
                if j == i:
                    bin_name = zbin_richbin

        cat = treecorr.Catalog(
            self.get_input("binned_lens_catalog"),
            ext=f"/cluster_bin/{bin_name}",
            ra_col="ra",
            dec_col="dec",
            #w_col="weight",
            ra_units="degree",
            dec_units="degree",
            patch_centers=None,#self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_lens_catalog", i),
        )
        return cat

    def get_random_catalog(self, i):
        # As with the lens catalog version, we add the r_col keyword
        # compare to the parent class
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
            patch_centers=None,#self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_random_catalog", i),
        )
        return rancat

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
            patch_centers=None,#self.get_input("patch_centers"),
            save_patch_dir=self.get_patch_dir("binned_shear_catalog", i),
            flip_g1=self.config["flip_g1"],
            flip_g2=self.config["flip_g2"],
        )
        
        return cat
    
    def call_treecorr(self, i, j, k):
        # The parent class uses the label for gamma_t measurements
        # here, so we overwrite it.
        result = super().call_treecorr(i, j, k)
        # choose a better name maybe!
        return result._replace(corr_type="galaxy_shearDensity_rlens")

    # def write_output(self, source_list, lens_list, meta, results):
    #     with self.open_output("cluster_profiles") as f:
    #         print("#i_bin  j_bin  mean_r_Mpc  rlens")
    #         #print("#i_bin  j_bin  mean_r_Mpc  rlens", file=f)
    #         for result in results:
    #             print(result.object)
    #             print(result.object.xi)
    #             for logr, xi in zip(result.object.meanlogr, result.object.xi):
    #                 print(result.i, result.j, np.exp(logr), xi)
    #                 #print(result.i, result.j, np.exp(logr), xi, file=f)
    # def write_output(self, source_list, lens_list, meta, results):
    #     import pickle
    #     import numpy as np
    #     # Dictionary to store results for each bin
    #     binned_cluster_stack = {}
    #     counts = None
    #     for result in results:
    #         for logr, xi in zip(result.object.meanlogr, result.object.xi):
    #                 print(result.i, result.j, np.exp(logr), xi)
    #         cluster_bin_edges = {}
    #         with self.open_input("binned_lens_catalog") as f:
    #             for j, zbin_richbin in enumerate(f['cluster_bin'].keys()):
    #                 bin_name = zbin_richbin
    #                 ext=f"/cluster_bin/{bin_name}",
    #                 metadata = f[ext].attrs
    #                 counts = len(list(f[ext]["cluster_id"]))
    #                 cluster_bin_edges["rich_max"] = metadata["rich_max"]
    #                 cluster_bin_edges["rich_min"] = metadata["rich_min"]
    #                 cluster_bin_edges["z_max"] = metadata["z_max"]
    #                 cluster_bin_edges["z_min"] = metadata["z_min"]
    #         # Use (i_bin, j_bin) as key
    #         key = (result.i, result.j)
    
    #         # Save radial bins
    #         radial_bins = np.exp(result.object.meanlogr)
    #         xi = result.object.xi
    
    #         # Save metadata for this bin, if available
    #         bin_meta = meta.get(key, {})
    
    #         # Store the data
    #         binned_cluster_stack[key] = {
    #             "cluster_rich_bin_edges": cluster_bin_edges,
    #             "n_cl": redshift_edges,
    #             "radial_bins": radial_bins,
    #             "xi": xi,
    #             "counts": counts
    #         }

    #     # Save the dictionary as a pickle file
    #     output_file = self.get_output("cluster_profiles")
    #     with open(output_file, "wb") as f:
    #         pickle.dump(binned_cluster_stack, f)
    
    #     print(f"TreeCorr results saved in pickle format to {output_file}")


    def write_output(self, source_list, lens_list, meta, results):
        import pickle
        import numpy as np
    
        # Dictionary to store stacked results per lens bin (richness-redshift bin)
        binned_cluster_stack = {}
    
        # --- Precompute string keys for all lens bins ---
        lens_keys = []
        with self.open_input("binned_lens_catalog") as f:
            for j, bin_name in enumerate(f["cluster_bin"].keys()):
                lens_keys.append(bin_name)
    
        # --- Aggregate results over all source bins ---
        for result in results:
            j = result.j  # lens bin index
            key = lens_keys[j]  # use string key
            radial_bins = np.exp(result.object.meanlogr)
            xi = result.object.xi
            npairs = result.object.npairs
    
            # TreeCorr covariance for this source-lens pair
            try:
                cov = result.object.cov['tan_sc']
            except Exception:
                cov = np.zeros((len(xi), len(xi)))
    
            if key not in binned_cluster_stack:
                # Initialize with first source bin
                binned_cluster_stack[key] = {
                    "radial_bins": radial_bins,
                    "xi_sum": xi * npairs,       # weighted sum
                    "npairs_sum": npairs.copy(),  # total pairs
                    "cov_sum": cov * npairs[:, None]  # weighted sum of covariance
                }
            else:
                # Aggregate over source bins
                binned_cluster_stack[key]["xi_sum"] += xi * npairs
                binned_cluster_stack[key]["npairs_sum"] += npairs
                binned_cluster_stack[key]["cov_sum"] += cov * npairs[:, None]
    
        # --- Compute the weighted average xi and covariance for each lens bin ---
        for d in binned_cluster_stack.values():
            # xi
            with np.errstate(invalid='ignore', divide='ignore'):
                d["xi"] = np.divide(
                    d["xi_sum"], d["npairs_sum"],
                    out=np.full_like(d["xi_sum"], np.nan),
                    where=d["npairs_sum"] != 0
                )
            # covariance (diagonal only)
            d["cov"] = np.divide(
                d["cov_sum"], d["npairs_sum"][:, None],
                out=np.full_like(d["cov_sum"], np.nan),
                where=d["npairs_sum"][:, None] != 0
            )
    
            del d["xi_sum"], d["npairs_sum"], d["cov_sum"]
    
        # --- Add cluster metadata from the lens catalog ---
        with self.open_input("binned_lens_catalog") as f:
            for j, bin_name in enumerate(f["cluster_bin"].keys()):
                attrs = f[f"/cluster_bin/{bin_name}"].attrs
                counts = len(list(f[f"/cluster_bin/{bin_name}"]["cluster_id"]))
                key = lens_keys[j]
                if key in binned_cluster_stack:
                    binned_cluster_stack[key]["cluster_bin_edges"] = {
                        "rich_min": attrs["rich_min"],
                        "rich_max": attrs["rich_max"],
                        "z_min": attrs["z_min"],
                        "z_max": attrs["z_max"]
                    }
                    binned_cluster_stack[key]["n_cl"] = counts
    
        # --- Save the stacked results as a pickle file ---
        output_file = self.get_output("cluster_profiles")
        with open(output_file, "wb") as f:
            pickle.dump(binned_cluster_stack, f)
    
        print(f"TreeCorr results saved in pickle format to {output_file}")

        
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
                nbin_lens = len(f["cluster_bin"])

        source_list = list(range(nbin_source))
        lens_list = list(range(nbin_lens))

        return source_list, lens_list
        
    def select_calculations(self, source_list, lens_list):
        calcs = []
    
        if self.config["do_shear_pos"]:
            print("DOING SHEAR-POS")
            k = SHEAR_POS
    
            # --- Load source bin edges from shear_tomography_catalog ---
            with self.open_input("shear_tomography_catalog") as f:
                tomo = f["tomography"]
                nbin_source = tomo.attrs["nbin"]
                source_zmin = [tomo.attrs[f"zmin_{i}"] for i in range(nbin_source)]
                source_zmax = [tomo.attrs[f"zmax_{i}"] for i in range(nbin_source)]
    
            # --- Loop over source and lens bins ---
            for i in source_list:
                for j in lens_list:
                    # --- Load lens bin z-range ---
                    with self.open_input("binned_lens_catalog") as f:
                        bin_name = list(f["cluster_bin"])[j]
                        attrs_bin = f[f"cluster_bin/{bin_name}"].attrs
                        lens_zmin, lens_zmax = attrs_bin["z_min"], attrs_bin["z_max"]
    
                    # --- Check for redshift overlap ---
                    overlap = not (
                        lens_zmax < source_zmin[i] or lens_zmin > source_zmax[i]
                    )
    
                    if overlap:
                        print(
                            f"Skipping lens bin {j} (z=[{lens_zmin:.3f}, {lens_zmax:.3f}]) "
                            f"and source bin {i} (z=[{source_zmin[i]:.3f}, {source_zmax[i]:.3f}]) — overlap detected."
                        )
                        continue
                    calcs.append((i, j, k))

        if self.rank == 0:
            print(f"Running {len(calcs)} calculations: {calcs}")

        return calcs
