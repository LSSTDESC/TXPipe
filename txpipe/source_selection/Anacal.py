from .base import TXSourceSelectorBase
from .base import select_weak_lensing_sample, select_tomographic_weak_lensing_sample
from ..shear_calibration import AnaCalCalculator, band_variants
import numpy as np
from ceci.config import StageParameter


class TXSourceSelectorAnacal(TXSourceSelectorBase):
    """
    Source selection and tomography for AnaCal catalogs

    This selector subcallss is designed for anacal-type catalogs like those
    that DESC plans to produce for Ruin data.
    """

    name = "TXSourceSelectorAnaCal"

    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": StageParameter(
            float, 
            required=True,
            msg= "Delta gmamma value for hte AnaCal response calculation"
        ),
    }

    def data_iterator(self):
        """
        This iterator returns chunks of data in dictionaries one by one.

        We call to a parent class method to do the main iteration; the work here is
        just choosing which columns to read.
        """
        
        bands = self.config["bands"]
        shear_cols = ["ra",
                      "dec",
                      "weight",
                      "mask_value",
                      "e1",
                      "e2",
                      "m00",
                      "m20",
                      "de1_dg1",
                      "de2_dg2",
                      "dm00_dg1",
                      "dm00_dg2",
                      "dm20_dg1",
                      "dm20_dg2"
                      ]
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="Anacal")

        if self.config["input_pz"]:
            shear_cols += ["mean_z"]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]

        chunk_rows = self.config["chunk_rows"]
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)
    
    def setup_output(self):
        """
        Prepare the output columns for the response values generated bby Anacal
        """
        outfile = super().setup_output()
        n = outfile["count/counts"].size
        group = outfile.create_group("response")
        group.create_dataset("R", (n, 1, 1), dtype="f")
        return outfile

