from .base import TXIngestCatalogFits
from ..data_types import ShearCatalog, FitsFile
from .dp1_details import (
    DP1_TRACTS,
    DP1_COSMOLOGY_TRACTS,
    ALL_TRACTS,
)
from ceci.config import StageParameter
import numpy as np
from ..utils.hdf_tools import h5py_shorten, repack
import warnings

# Suffixes on the merged catalog's photo-z point-estimate columns
# (zmode_0, zmode_1p, zmode_1m, zmode_2p, zmode_2m).  These become
# ``mean_z`` / ``mean_z_{1p,1m,2p,2m}`` in the ingested shear catalog
# and are consumed by TXSourceSelectorAnaCal for the tomographic
# bin-migration term of R_sel via _DataWrapper suffix lookup.
PZ_SUFFIXES = ("0", "1p", "1m", "2p", "2m")


class TXIngestAnacal(TXIngestCatalogFits):
    """
    Ingestion of an anacal catalog, generated from actual Rubin data. This
    stage, will take an anacal catalog, from either the butler, or a file
    (parquet), and ingest it into TXPipe format (HDF5).
    """

    name = "TXIngestAnacal"
    inputs = [
        ("anacal_catalog", FitsFile)
    ]
    outputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options = {
        "use_butler": StageParameter(
            bool, True,
            msg="Should be left on, unless you got an external file, "
                "in that case knock yourself out!",
        ),
        "butler_config_file": StageParameter(
            str,
            "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
            msg="Path to the LSST butler config file.",
        ),
        "butler_object_name": StageParameter(
            str, "deep_coadd_cell_anacal_merged",
        ),
        "cosmology_tracts_only": StageParameter(
            bool, True, msg="Use only cosmology tracts.",
        ),
        "select_field": StageParameter(
            str, "",
            msg="Field to select (overrides cosmology_tracts_only).",
        ),
        "select_tracts": StageParameter(
            list, [],
            msg="list of tracts (overrides cosmology_tracts_only, but "
                "not select_field).",
        ),
        "collections": StageParameter(
            str, "LSSTComCam/DP1", msg="Butler collections to use.",
        ),
        "tracts": StageParameter(
            str, "",
            msg="Comma-separated list of tracts to use (empty for all).",
        ),
        "prefix": StageParameter(
            str, "fpfs",
            msg="prefix indicating the method used to calculate the ",
        ),
        "bands": StageParameter(
            list, ["g", "r", "i", "z", "y"], msg="string of flux bands",
        ),
        "scale": StageParameter(
            str, "gauss2",
            msg="scale radius for the convolution with Gaussian PSF",
        ),
    }

    def run(self):
        if self.config["use_butler"]:
            self.butler_run()
        else:
            warnings.warn("File run is a depricated method and might not function correctly.")
            self.file_run()

        print("repacking files")
        repack(self.get_output("shear_catalog"))

    def butler_run(self):
        error_msg = (
            "The LSST Science Pipelines are not installed in this environment, "
            "or are not configured correctly to access the data. "
            "See the note in the file example/dp1/ingest.yml for how to set "
            "this up on NERSC."
        )
        try:
            from lsst.daf.butler import Butler
        except Exception as e:
            raise ImportError(error_msg) from e

        # Configure and create the butler. There are several ways to do this,
        # Here we use a central collective butler yaml file from NERSC.

        butler_config_file = self.config["butler_config_file"]
        collections = self.config["collections"]
        error_msg2 = error_msg + (
            ' Or there is a typo in the collection you have looked for.'
        )
        try:
            butler = Butler(butler_config_file, collections=collections)
        except Exception as e:
            raise RuntimeError(error_msg2) from e

        if self.config["select_field"]:
            tracts = DP1_TRACTS[self.config["select_field"]]
        elif self.config["select_tracts"]:
            tracts = self.config["select_tracts"]
        elif self.config["cosmology_tracts_only"]:
            tracts = DP1_COSMOLOGY_TRACTS
        else:
            tracts = ALL_TRACTS

        object_name = self.config["butler_object_name"]
        n = self.get_catalog_size(butler, object_name)

        created_files = False
        data_set_refs = butler.query_datasets(object_name)
        n_chunks = len(data_set_refs)

        shear_start = 0
        for i, ref in enumerate(data_set_refs):
            tract = ref.dataId["tract"]
            if tract not in tracts:
                print(
                    f"Skipping chunk {i + 1} / {n_chunks} since tract "
                    f"{tract} is not selected"
                )
                continue

            d = butler.get(object_name,
                           dataId=ref.dataId,
                           )
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"Skipping chunk {i + 1} / {n_chunks} since it is empty")
                continue

            shear_data = self.process_anacal_shear_data(d)
            if not created_files:
                created_files = True
                shear_outfile = self.setup_output(
                    "shear_catalog", "shear", shear_data, n,
                )
                shear_outfile["shear"].attrs["catalog_type"] = "anacal"

            shear_end = shear_start + len(shear_data["ra"])
            self.write_output(
                shear_outfile, "shear", shear_data, shear_start, shear_end,
            )

            print(
                f"Processing chunk {i + 1} / {n_chunks} into rows "
                f"{shear_start:,} - {shear_end:,}"
            )
            shear_start = shear_end

        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, shear_end)
        self.aliasing(shear_outfile)
        shear_outfile.close()

    def file_run(self):
        n, dtypes = self.get_meta("anacal_catalog")

        prefix = self.config["prefix"]

        file = self.open_input("anacal_catalog")
        data = file[1]
        if not hasattr(data, "colnames"):
            warnings.warn(
                f"file_run's input ({type(data)}) has no `.colnames` attribute;"
                "process_anacal_shear_data exp[ects an astropy Table like butler_run "
                "provides, so this will likely fail."
            )

        shear_data = self.process_anacal_shear_data(data)

        shear_outfile = self.setup_output(
            "shear_catalog", "shear", shear_data, n,
        )
        shear_outfile["shear"].attrs["catalog_type"] = "anacal"

        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, len(shear_data["ra"]))

        self.aliasing(shear_outfile)
        shear_outfile.close()

    def setup_input(self):
        prefix = self.config["prefix"]
        scale = self.config["scale"]
        cols = [
            "ra",
            "dec",
            "wsel",
            "mask_value",
            f"{prefix}_e1",
            f"{prefix}_e2",
            f"{prefix}_m00",
            f"{prefix}_m20",
        ]
        cols += ["dwsel" + suffix for suffix in ["_dg1", "_dg2"]]
        cols += [
            prefix + delta + suffix
            for delta in ["_de1", "_de2", "_dm00", "_dm20"]
            for suffix in ["_dg1", "_dg2"]
        ]
        bands = self.config["bands"]
        # i-band S/N + shear response, always from the fpfs1 family.
        # ``scale`` below only picks the flux used for magnitudes; the
        # brightness cut consumes the pre-computed fpfs1 S/N so it is
        # consistent regardless of the mag scale.
        cols += [
            "lsst_i_s2n_fpfs1",
            "lsst_i_ds2n_fpfs1_dg1",
            "lsst_i_ds2n_fpfs1_dg2",
        ]
        # Pre-computed AB magnitudes + shear responses on the DP1-v3
        # merged catalog (xlens.add_magnitude_columns at the fixed
        # MAG_ZERO_AB zeropoint) — passed through so downstream stages
        # can consume mags without redoing the nanojansky→mag math.
        # Both ``mag`` and its error ``mag_err`` carry shear responses.
        for b in bands:
            cols += [
                f"lsst_{b}_mag_{scale}",
                f"lsst_{b}_dmag_{scale}_dg1",
                f"lsst_{b}_dmag_{scale}_dg2",
                f"lsst_{b}_mag_{scale}_err",
                f"lsst_{b}_dmag_{scale}_err_dg1",
                f"lsst_{b}_dmag_{scale}_err_dg2",
            ]
        # Band-combined shape magnitude ``esq = e1^2 + e2^2`` and its
        # shear derivatives, emitted by xlens.MergePipe on the
        # WCS-corrected fpfs1 shape. Consumed by the |e|<emax cut in
        # TXSourceSelectorAnaCal.
        cols += ["esq", "desq_dg1", "desq_dg2"]

        # zmode_0 → mean_z; zmode_1p, zmode_1m, zmode_2p, zmode_2m → the
        # metacal-style shifted variants (built with dg=0.01 in xlens'
        # photoZPipe, so TXSourceSelectorAnaCal must use delta_gamma=0.01).
        cols += [f"zmode_{s}" for s in PZ_SUFFIXES]

        return cols

    def process_anacal_shear_data(self, data):
        bands = self.config["bands"]
        s = self.config["scale"]
        output = {name: data[name][:] for name in data.colnames}

        for band in bands:
            f = data[f"{band}_flux_{s}"][:]
            f_err = data[f"{band}_flux_{s}_err"][:]
            output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
            output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)

            if band == "i":
                output["s2n"] = f / f_err

            for d in ["dg1", "dg2"]:
                dd = data[f"{band}_dflux_{s}_"+d][:]
                output[f"mag_{band}_{d}"] = anacal_mag_response(f, dd)
                if band == "i":
                    output[f"ds2n_{d}"] = dd/f_err

        return output

    def setup_output(self, tag, group, first_chunk, n):
        f = self.open_output(tag)
        g = f.create_group(group)

        for name, col in first_chunk.items():
            g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f

    def write_output(self, outfile, group, data, start, end):
        g = outfile[group]
        for name, col in data.items():
            # replace masked values with nans
            if np.ma.isMaskedArray(col):
                col = col.filled(np.nan)
            g[name][start:end] = col

    def get_catalog_size(self, butler, dataset_type):
        import pyarrow.parquet

        n = 0
        for ref in butler.query_datasets(dataset_type):
            uri = butler.getURI(ref)
            if not uri.path.endswith(".parq"):
                raise ValueError(
                    f"Some data in dataset {dataset_type} was not in "
                    f"parquet format: {uri.path}"
                )
            with pyarrow.parquet.ParquetFile(uri.path) as f:
                n += f.metadata.num_rows
        return n

    def aliasing(self, outfile):
        prefix = self.config["prefix"]
        g = outfile["shear"]

        g["weight"] = g["wsel"]
        g["weight_dg1"] = g["dwsel_dg1"]
        g["weight_dg2"] = g["dwsel_dg2"]
        g["e1"] = g[f"{prefix}_e1"]
        g["e2"] = g[f"{prefix}_e2"]
        g["m00"] = g[f"{prefix}_m00"]
        g["m20"] = g[f"{prefix}_m20"]
        for delta in ["de1", "de2", "dm00", "dm20"]:
            g[f"{delta}_dg1"] = g[f"{prefix}_{delta}_dg1"]
            g[f"{delta}_dg2"] = g[f"{prefix}_{delta}_dg2"]
        
        g["mean_z"] = g["zmode_0"]
        for suf in ["1p", "1m", "2p", "2m"]:
            g[f"mean_z_{suf}"] = g[f"zmode_{suf}"]

