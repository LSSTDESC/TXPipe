from .base import TXSourceSelectorBase, BinStats
from .base import select_weak_lensing_sample, select_tomographic_weak_lensing_sample
from ..shear_calibration import (
    MetaDetectCalibrator,
    metadetect_variants,
    MetaDetectCalculator,
    band_variants,
    META_VARIANTS
)
import numpy as np
from ceci.config import StageParameter
from ..utils import rename_iterated


class TXSourceSelectorMetadetect(TXSourceSelectorBase):
    """
    Source selection and tomography for metadetect catalogs

    This subclass selects for MetaDetect catalogs, which is expected to be used for
    Rubin data. It computes the selection bias due to object detection by repeating
    the detection process under different applied shears.

    As a consequence the different calibration columns have different lengths, since
    different objects are detected in each case.
    """

    name = "TXSourceSelectorMetadetect"

    # add one option to the base class configuration
    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": StageParameter(
            float,
            required=True,
            msg="Delta gamma value for metadetect response calculation",
        ),
    }

    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        # Core quantities we need
        shear_cols = metadetect_variants("T", "s2n", "g1", "g2", "ra", "dec", "mcal_psf_T_mean", "weight", "flags")

        # Magnitudes and errors
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="metadetect")
        renames = {}

        # We need truth shears and/or PZ point-estimates for each shear too
        if self.config["input_pz"]:
            shear_cols += metadetect_variants("mean_z")
        elif self.config["true_z"]:
            shear_cols += metadetect_variants("redshift_true")

        for prefix in ["00", "1p", "1m", "2p", "2m"]:
            renames[f"{prefix}/mcal_psf_T_mean"] = f"{prefix}/psf_T_mean"

        # This is a parent ceci.PipelineStage method.
        # It returns an iterator we loop through.
        # The "longest=True" option means that the iterator will
        # continue looping even when some of the columns have been exhausted, which is 
        # what we want here since the different shear variants have different lengths.
        # The calibration calculation needs to deal with this.
        it = self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows, longest=True)
        return rename_iterated(it, renames)

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [MetaDetectCalculator(select_tomographic_weak_lensing_sample, delta_gamma) for i in range(nbin_source)]
        calculators.append(MetaDetectCalculator(select_weak_lensing_sample, delta_gamma))
        return calculators

    def accumulate_statistics(self, output_file, shear_data, start, end, tomo_bin, per_object_response, number_density_stats):
        # Save the tomography for this chunk
        self.write_tomography(output_file, start, end, tomo_bin, per_object_response)

        # Accumulate information on the number counts and the selection biases.
        # These will be brought together at the end.
        # For metadetect we only want to accumulate info for the 00
        # variant, so we take the first tomo_bin value.
        number_density_stats.add_data(shear_data, tomo_bin[0])

    def write_tomography(self, outfile, start, end, source_bin, per_object_response):
        # Write out each of the individual variants.
        # The basic "bin" column was set up to be the same as the 00 variant,
        # so we can just write to all of them.
        for i, v in enumerate(META_VARIANTS):
            outfile[f"tomography/bin_{v}"][start:end] = source_bin[i]

        assert per_object_response is None, "MetaDetect does not produce per-object response values, only per-bin values, so this should be None"


    def apply_simple_redshift_cut(self, data):

        # Otherwise we have to do it once for each variant
        pz_data = {}
        variants = ["00/", "1p/", "2p/", "1m/", "2m/"]
        for v in variants:
            if self.config["true_z"]:
                zz = data[f"{v}redshift_true"]
            else:
                zz = data[f"{v}mean_z"]

            pz_data_v = np.zeros(len(zz), dtype=int) - 1
            for zi in range(len(self.config["source_zbin_edges"]) - 1):
                mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (
                    zz < self.config["source_zbin_edges"][zi + 1]
                )
                pz_data_v[mask_zbin] = zi

            pz_data[f"{v}zbin"] = pz_data_v

        return pz_data

    def setup_output(self):
        """
        MetaDetect outputs do not include per-object calibration values,
        only the per-bin values.
        """
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()

        # For the metadetect we also want to save the selected bin for every variant.
        # We will need this later on in the pipeline for diagnostics.
        # We do the 00 variant separately because it's just the same as the base
        # case that was set up in the parent class call above, so we just link to that
        # dataset rather than creating a new one.
        with self.open_input("shear_catalog") as infile:
            for v in META_VARIANTS[1:]:
                n = infile[f"shear/{v}/ra"].size
                outfile["tomography"].create_dataset(f"bin_{v}", (n,), dtype=np.int32)
        # Link the 00 variant to the base tomography/bin dataset 
        outfile["tomography/bin_00"] = outfile["tomography/bin"]

        # There is only global calibration information for metadetect, nothing
        # per-bin.
        nbin_source = outfile["counts/counts"].size
        group = outfile.create_group("response")
        # Per-bin 2x2 calibration matrix
        group.create_dataset("R", (nbin_source, 2, 2), dtype="f")
        # Global calibration matrix
        group.create_dataset("R_2d", (2, 2), dtype="f")
        return outfile

    def compute_output_stats(self, calculator, mean, variance):
        # Collate calibration values
        R, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = MetaDetectCalibrator(R, mean, mu_is_calibrated=False)
        mean_e = calibrator.mu.copy()

        # Apply to the variances to get sigma_e
        P = np.diag(np.linalg.inv(R @ R))
        sigma_e = np.sqrt(0.5 * P @ variance)

        # Like metacal, N_eff = N for metadetect
        return BinStats(N, Neff, mean_e, sigma_e, calibrator)

    def calculate_tomography(self, pz_data, shear_data, calculators):
        """
        Select objects to go in each tomographic bin and their calibration.

        Parameters
        ----------

        pz_data: table or dict of arrays
            A chunk of input photo-z data containing mean values for each object
        shear_data: table or dict of arrays
            A chunk of input shear data with metacalibration variants.
        """
        nbin = len(self.config["source_zbin_edges"]) - 1
        n = len(list(shear_data.values())[0])

        tomo_bins = []
        for v in META_VARIANTS:
            n = len(shear_data[f"{v}/g1"])
            tomo_bin = np.repeat(-1, n)
            tomo_bins.append(tomo_bin)

        data = {**pz_data, **shear_data}

        R = self.compute_per_object_response(data)

        for i in range(nbin):
            selections = calculators[i].add_data(data, self.config, i)
            for j, v in enumerate(META_VARIANTS):
                tomo_bins[j][selections[j]] = i

        # and calibrate the 2D sample.
        # This calibrator refers to select_weak_lensing_sample
        calculators[-1].add_data(data, self.config)

        return tomo_bins, R
