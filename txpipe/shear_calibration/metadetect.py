from .base import TXComputeShearCalibrationBase
from ..calibration_utils import (
    metadetect_variants,
    MetaDetectCalculator,
    band_variants,
    META_VARIANTS
)
import numpy as np
from ceci.config import StageParameter
from ..utils import rename_iterated


class TXComputeShearCalibrationMetadetect(TXComputeShearCalibrationBase):
    """
    Source selection and tomography for metadetect catalogs

    This subclass selects for MetaDetect catalogs, which is expected to be used for
    Rubin data. It computes the selection bias due to object detection by repeating
    the detection process under different applied shears.

    As a consequence the different calibration columns have different lengths, since
    different objects are detected in each case.
    """

    name = "TXComputeShearCalibrationMetadetect"

    # add one option to the base class configuration
    config_options = {
        **TXComputeShearCalibrationBase.config_options,
        "delta_gamma": StageParameter(
            float,
            required=True,
            msg="Delta gamma value for metadetect response calculation",
        ),
    }

    # We always refer to the meta variants using this so that
    # we can subclass to do the scalar version that just has 1p and 1m
    variants = META_VARIANTS


    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [
            MetaDetectCalculator(self.select, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(MetaDetectCalculator(self.select_any_bin, delta_gamma))
        return calculators

    def write_tomography(self, outfile, start, end, source_bin, per_object_response):
        # The per_object_response is not really that - see below in the accumulate_selection
        # function. It actually just tells us which variant we are currently processing
        variant = per_object_response
        outfile[f"tomography/{variant}/bin"][start:end] = source_bin


    def setup_output(self, nbin_source):
        """
        MetaDetect outputs do not include per-object calibration values,
        only the per-bin values.
        """
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output(nbin_source)

        # For the metadetect we also want to save the selected bin for every variant.
        # We will need this later on in the pipeline for diagnostics.
        # We do the 00 variant separately because it's just the same as the base
        # case that was set up in the parent class call above, so we just link to that
        # dataset rather than creating a new one.
        with self.open_input("shear_catalog") as infile:
            for v in self.variants[1:]:
                n = infile[f"shear/{v}/ra"].size
                outfile["tomography"].create_dataset(f"{v}/bin", (n,), dtype=np.int16)
        # Link the 00 variant to the base tomography/bin dataset 
        outfile["tomography/ns/bin"] = outfile["tomography/bin"]
        self.setup_response_output(outfile, nbin_source)

        return outfile

    def setup_response_output(self, outfile, nbin_source):
        # There is only global calibration information for metadetect, nothing
        # per-bin.
        group = outfile.create_group("response")
        # Per-bin 2x2 calibration matrix
        group.create_dataset("R", (nbin_source, 2, 2), dtype="f")
        # Global calibration matrix
        group.create_dataset("R_2d", (2, 2), dtype="f")



    def accumulate_selection(self, data, calculators):
        """
        Select objects to go in each tomographic bin and their calibration.

        Parameters
        ----------

        pz_data: table or dict of arrays
            A chunk of input photo-z data containing mean values for each object
        shear_data: table or dict of arrays
            A chunk of input shear data with metacalibration variants.
        """
        nbin = self.config["nbin_source"]
        n = len(list(data.values())[0])

        # The data in here will just be for a single tomo bin.
        # Figure out which one. It should have bin_{v} in for one variant
        for i, v in enumerate(self.variants):
            if f'{v}/bin' in data:
                variant = v
                variant_index = i
                break
        else:
            raise ValueError("Could not determine what was in this dictionary")

        # Filled in below
        tomo_bin = np.repeat(-1, n)

        # There is no per-object response value in metadatect, but rather
        # than re-write things more sensibly I am going to use this variable,
        # which is just going to be passed back to us in write_tomography above,
        # to store the variant we are running on.
        not_really_per_object_response = variant

        for i in range(nbin):
            selections = calculators[i].add_data(data, i)
            # For compatibilty with other uses of the calculator,
            # like metacal, `selections` here will be for every variant,
            # even though we only actually supplied one. The other
            # variant selections will be empty.
            selection = selections[variant_index]
            tomo_bin[selection] = i

        # and calibrate the 2D sample.
        calculators[-1].add_data(data)

        return tomo_bin, not_really_per_object_response

    def column_groups(self):
        # for metadetect
        return [
            {
                "shear_catalog": [f"{v}/g1", f"{v}/g2", f"{v}/weight"],
                "shear_bin_assignments": [f"{v}/bin"],
                "shear_selection": [f"{v}/selection"],
            }
            for v in self.variants
        ]
