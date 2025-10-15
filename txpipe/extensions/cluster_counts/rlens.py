from ...twopoint import TXTwoPoint
from ...data_types import (
    HDFFile,
    ShearCatalog,
    PNGFile,
    TextFile,
)
import numpy as np

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

class TXTwoPointRLens(TXTwoPoint):
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
    name = "TXTwoPointRLens"
    inputs = [
        ("binned_lens_catalog", HDFFile),
        ("binned_shear_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("patch_centers", TextFile),
    ]
    outputs = [
        ("rlens_measurement", TextFile),
        ("cluster_shear_catalogs", HDFFile),           
    ]

    config_options = {
        # TODO: Allow more fine-grained selection of 2pt subsets to compute
        "calcs": [0, 1, 2],
        "min_sep": 1.0,  # Megaparsec
        "max_sep": 50.0,  # Megaparsec
        "nbins": 9,
        "bin_slop": 0.1,
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
        "metric": "Rlens",
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
            #r_col="comoving_distance",
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
            #r_col="comoving_distance",
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

    def write_output(self, source_list, lens_list, meta, results):
        with self.open_output("rlens_measurement") as f:
            print("#i_bin  j_bin  mean_r_Mpc  rlens")
            print("#i_bin  j_bin  mean_r_Mpc  rlens", file=f)
            for result in results:
                for logr, xi in zip(result.object.meanlogr, result.object.xi):
                    print(result.i, result.j, np.exp(logr), xi)
                    print(result.i, result.j, np.exp(logr), xi, file=f)

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

