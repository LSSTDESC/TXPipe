from .base import TXSourceSelectorBase
from .base import select_weak_lensing_sample, select_tomographic_weak_lensing_sample
from ..shear_calibration import (
    scalar_metadetect_variants,
    ScalarMetaDetectCalculator,
    band_variants,
    SCALAR_META_VARIANTS,
)
import numpy as np
from ceci.config import StageParameter


class TXSourceSelectorScalarMetadetect(TXSourceSelectorBase):
    """
    Source selection and tomography for scalar metadetect catalogs.

    This subclass selects for scalar MetaDetect catalogs, which keep only the
    ns, 1p, and 1m variants and use a scalar response model.
    """

    name = "TXSourceSelectorScalarMetadetect"

    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": StageParameter(
            float,
            required=True,
            msg="Delta gamma value for scalar metadetect response calculation",
        ),
    }

    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        shear_cols = scalar_metadetect_variants(
            "T",
            "s2n",
            "g1",
            "g2",
            "ra",
            "dec",
            "weight",
            "psf_T_mean",
            "flags",
        )

        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="scalar_metadetect")

        if self.config["input_pz"]:
            shear_cols += scalar_metadetect_variants("mean_z")
        elif self.config["true_z"]:
            shear_cols += scalar_metadetect_variants("redshift_true")

        it = self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows, longest=True)
        return it

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [
            ScalarMetaDetectCalculator(select_tomographic_weak_lensing_sample, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(ScalarMetaDetectCalculator(select_weak_lensing_sample, delta_gamma))
        return calculators

    def write_tomography(self, outfile, start, end, source_bin, per_object_response):
        # The stated start and end values are not relevant here as they are the global
        # start and end index assuming a single block catalog.
        for i, v in enumerate(SCALAR_META_VARIANTS):
            start = self.current_bin_output_index[v]
            end = start + source_bin[i].size
            outfile[f"tomography/bin_{v}"][start:end] = source_bin[i]
            self.current_bin_output_index[v] = end

        assert per_object_response is None, (
            "ScalarMetaDetect does not produce per-object response values, only per-bin values, "
            "so this should be None"
        )

    def apply_simple_redshift_cut(self, data):
        pz_data = {}
        variants = SCALAR_META_VARIANTS
        for v in variants:
            if self.config["true_z"]:
                zz = data[f"{v}/redshift_true"]
            else:
                zz = data[f"{v}/mean_z"]

            pz_data_v = np.zeros(len(zz), dtype=int) - 1
            for zi in range(len(self.config["source_zbin_edges"]) - 1):
                mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (
                    zz < self.config["source_zbin_edges"][zi + 1]
                )
                pz_data_v[mask_zbin] = zi

            pz_data[f"{v}/zbin"] = pz_data_v

        return pz_data

    def apply_no_tomography_cut(self, shear_data):
        pz_data = {}
        variants = SCALAR_META_VARIANTS
        for v in variants:
            pz_data[f"{v}/zbin"] = np.zeros(shear_data[f"{v}/ra"].size, dtype=int)
        return pz_data


    def setup_output(self, nbin_source):
        outfile = super().setup_output(nbin_source)

        with self.open_input("shear_catalog") as infile:
            for v in SCALAR_META_VARIANTS[1:]:
                n = infile[f"shear/{v}/ra"].size
                outfile["tomography"].create_dataset(f"bin_{v}", (n,), dtype=np.int32)

        outfile["tomography/bin_ns"] = outfile["tomography/bin"]

        group = outfile.create_group("response")
        group.create_dataset("R", (nbin_source,), dtype="f")
        group.create_dataset("R_2d", (1,), dtype="f")

        self.current_bin_output_index = {v: 0 for v in SCALAR_META_VARIANTS}
        return outfile

    def calculate_tomography(self, pz_data, shear_data, calculators):
        nbin = self.config["nbin_source"]
        n = len(list(shear_data.values())[0])

        tomo_bins = []
        for v in SCALAR_META_VARIANTS:
            n = len(shear_data[f"{v}/g1"])
            tomo_bin = np.repeat(-1, n)
            tomo_bins.append(tomo_bin)

        data = {**pz_data, **shear_data}

        R = self.compute_per_object_response(data)

        for i in range(nbin):
            selections = calculators[i].add_data(data, self.config, i)
            for j, v in enumerate(SCALAR_META_VARIANTS):
                tomo_bins[j][selections[j]] = i

        calculators[-1].add_data(data, self.config)

        return tomo_bins, R
