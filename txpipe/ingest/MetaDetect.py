from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection
from .lsst import process_metadetect_data
from ceci.config import StageParameter
from ..utils.hdf_tools import h5py_shorten, repack
from ..utils.splitters import MetaDetectSplitter
import numpy as np
import os
import pyarrow.parquet as pq

# The tract values are listed in table 2 of that paper:
DP1_COSMOLOGY_FIELDS = [
    "EDFS",
    "ECDFS",
    "LGLF",
]


DP1_TRACTS = {
    # Euclid Deep Field South
    "EDFS": [2393, 2234, 2235, 2394],
    # Extended Chandra Deep Field South
    "ECDFS": [5062, 5063, 5064, 4848, 4849],
    # Low Galactic Latitude Field / Rubin_SV_095_-25
    "LGLF": [5305, 5306, 5525, 5526],
    # Fornax Dwarf Spheroidal Galaxy
    "FDSG": [4016, 4217, 4218, 4017],
    # Low Ecliptic Latitude Field / Rubin_SV_38_7
    "LELF": [10464, 10221, 10222, 10704, 10705, 10463],
    # Seagull Nebula
    "Seagull": [7850, 7849, 7610, 7611],
    # 47 Tuc Globular Cluster
    "47Tuc": [531, 532, 453, 454],
}

DP1_COSMOLOGY_TRACTS = sum([DP1_TRACTS[_field] for _field in DP1_COSMOLOGY_FIELDS], [])
ALL_TRACTS = sum(DP1_TRACTS.values(), [])


# In case useful later:
DP1_FIELD_CENTERS = {
    "47 Tuc Globular Cluster": (6.02, -72.08),
    "Low Ecliptic Latitude Field": (37.86, 6.98),
    "Fornax Dwarf Spheroidal Galaxy": (40.00, -34.45),
    "Extended Chandra Deep Field South": (53.13, -28.10),
    "Euclid Deep Field South": (59.10, -48.73),
    "Low Galactic Latitude Field": (95.00, -25.00),
    "Seagull Nebula": (106.23, -10.51),
}


DP1_SURVEY_PROPERTIES = {
    "deepCoadd_exposure_time_consolidated_map_sum": "Total exposure time accumulated per sky position (second)",
    "deepCoadd_epoch_consolidated_map_min": "Earliest observation epoch (MJD)",
    "deepCoadd_epoch_consolidated_map_max": "Latest observation epoch (MJD)",
    "deepCoadd_epoch_consolidated_map_mean": "Mean observation epoch (MJD)",
    "deepCoadd_psf_size_consolidated_map_weighted_mean": "Weighted mean of PSF characteristic width as computed from the determinant radius (pixel)",
    "deepCoadd_psf_e1_consolidated_map_weighted_mean": "Weighted mean of PSF ellipticity component e1",
    "deepCoadd_psf_e2_consolidated_map_weighted_mean": "Weighted mean of PSF ellipticity component e2",
    "deepCoadd_psf_maglim_consolidated_map_weighted_mean": "Weighted mean of PSF flux 5σ magnitude limit (magAB)",
    "deepCoadd_sky_background_consolidated_map_weighted_mean": "Weighted mean of background light level from the sky (nJy)",
    "deepCoadd_sky_noise_consolidated_map_weighted_mean": "Weighted mean of standard deviation of the sky level (nJy)",
    "deepCoadd_dcr_dra_consolidated_map_weighted_mean": "Weighted mean of DCR-induced astrometric shift in right ascension direction, expressed as a proportionality factor",
    "deepCoadd_dcr_ddec_consolidated_map_weighted_mean": "Weighted mean of DCR-induced astrometric shift in declination direction, expressed as a proportionality factor",
    "deepCoadd_dcr_e1_consolidated_map_weighted_mean": "Weighted mean of DCR-induced change in PSF ellipticity (e1), expressed as a proportionality factor",
    "deepCoadd_dcr_e2_consolidated_map_weighted_mean": "Weighted mean of DCR-induced change in PSF ellipticity (e2), expressed as a proportionality factor",
}


class TXIngestMetaDetect(PipelineStage):
    """
    Initial ingestion of the Rubin MetaDetect catalog
    """

    name = "TXIngestMetaDetect"
    inputs = []
    outputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options= {
        "use_butler": StageParameter(
                                     bool,
                                     True,
                                     msg="Should be left on, unless you got an external file, in that case knock yourself out!"),
        "butler_config_file": StageParameter(
            str, 
            "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
            msg="Path to the LSST butler config file."
        ),
        "cosmology_tracts_only": StageParameter(bool, True, msg="Use only cosmology tracts."),
        "select_field": StageParameter(str, "", msg="Field to select (overrides cosmology_tracts_only)."),
        "collections": StageParameter(str, "LSSTComCam/DP1", msg="Butler collections to use."),
        "file_path": StageParameter(str, None, msg="if not using a Butler, you need to give a path to the file.")
    }

    def run(self):
        if self.config["use_butler"]:
            self.butler_run()
        else:
            self.file_run()

        # Run h5repack on the file
        print("Repacking files")
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
        except:
            raise ImportError(error_msg)
        

        # Configure and create the butler. There are several ways to do this,
        # Here we use a central collective butler yaml file from NERSC.

        butler_config_file = self.config["butler_config_file"]
        collections = self.config["collections"]
        try:
            butler = Butler(butler_config_file, collections=collections)
        except:
            raise RuntimeError(error_msg)

        if self.config["select_field"]:
            tracts = DP1_TRACTS[self.config["select_field"]]
        elif self.config["cosmology_tracts_only"]:
            tracts = DP1_COSMOLOGY_TRACTS
        else:
            tracts = ALL_TRACTS

        #n = self.get_catalog_size(butler, "ShearObject")
        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"

        created_files = False
        data_set_refs = butler.query_datasets('object_shear_all')
        n_chunks = len(data_set_refs)
        input_columns = self.get_input_columns()

        for i, ref in enumerate(data_set_refs):
            tract = ref.dataId["tract"]
            if tract not in tracts:
                print(f"Skipping chunk {i + 1} / {n_chunks} since tract {tract} is not selected")
                continue

            d = butler.get('object_shear_all',
                           dataId=ref.dataId,
                           parameters={"columns": input_columns}
                           )
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"Skipping chunk {i + 1} / {n_chunks} since it is empty")
                continue

            shear_data = process_metadetect_data(d)
            if not created_files:
                created_files = True
                variants = {
                    "ns": len(shear_data["ns"]),
                    "1p": len(shear_data["1p"]),
                    "1m": len(shear_data["1m"]),
                    "2p": len(shear_data["2p"]),
                    "2m": len(shear_data["2m"]),
                    }
                columns = list(shear_data["ns"].keys())
                dtypes = {key: shear_data["ns"][key].dtype for key in shear_data["ns"]}
                splitter = MetaDetectSplitter(group, columns, variants, dtypes=dtypes)

            for variant in ["ns", "1p", "1m", "2p", "2m"]:
                splitter.write_bin(shear_data[variant], variant)
            print(f"Processing chunk {i + 1} / {n_chunks}")

        splitter.finish()
        shear_outfile.close()

    def file_run(self):
        file_path = self.config("file_path")
        if file_path == None:
            raise RuntimeError("You must either use a butler, or specify a file_path to your metadetect catalog.")

        if not os.path.exists(file_path):
            raise RuntimeError("No file where you said it would be.")

        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"

        created_files = False
        input_columns = self.get_input_columns()

        chunk_size = self.config("chunk_rows")
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(columns=input_columns, batch_size=chunk_size):
            shear_data = process_metadetect_data(batch)
            
            if not created_files:
                created_files = True
                variants = {
                    "ns": len(shear_data["ns"]),
                    "1p": len(shear_data["1p"]),
                    "1m": len(shear_data["1m"]),
                    "2p": len(shear_data["2p"]),
                    "2m": len(shear_data["2m"]),
                    }
                columns = list(shear_data["ns"].keys())
                splitter = MetaDetectSplitter(group, columns, variants)

            for variant in ["ns", "1p", "1m", "2p", "2m"]:
                data = sanitize(shear_data[variant])
                splitter.write_bin(data, variant)

        splitter.finish()
        shear_outfile.close()

    def get_input_columns(self):
        input_columns = [
            "shearObjectId",
            #"cellId", #removed
            'cell_x',
            'cell_y',
            "metaStep",
            "ra",
            "dec",
            #"mfrac",
            #"maskFractionCell", #Removed
            #"nEpochCell", #Removed
            "gauss_g1",
            "gauss_g2",
            "gauss_g1_g1_Cov",
            "gauss_g1_g2_Cov",
            "gauss_g2_g2_Cov",
            "gauss_T",
            "gauss_snr",
            "gauss_TErr",
            "gauss_psfReconvolved_g1", 
            "gauss_psfReconvolved_g2",
            'gauss_psfReconvolved_T',
            "psfOriginal_e1",
            "psfOriginal_e2",
            "psfOriginal_T",
            #Next follows the fluxes:
            "g_pgaussFlux",
            "r_pgaussFlux",
            "i_pgaussFlux",
            #"z_pgaussFlux",
            "g_pgaussFluxErr",
            "r_pgaussFluxErr",
            "i_pgaussFluxErr",
            #"z_pgaussFluxErr",
            "pgauss_T",
            "pgauss_TErr",
            #Various flags
            #"stamp_flags",
            "psfOriginal_flags",
            "gauss_psfReconvolved_flags",
            "gauss_object_flags",
            "g_gaussFlux_flags",
            "r_gaussFlux_flags",
            "i_gaussFlux_flags",
            #"z_gaussFlux_flags",
            "g_pgaussFlux_flags",
            "r_pgaussFlux_flags",
            "i_pgaussFlux_flags",
            #"z_pgaussFlux_flags",
            #"gauss_T_flags",
            #"pgauss_T_flags",
            "gauss_flags",
            "pgauss_flags",
            "gauss_shape_flags",
            
        ]
        return input_columns

    def setup_output(self, tag, group, first_chunk):
        f = self.open_output(tag)
        g = f.create_group(group)
        variants =  {
            "ns": len(first_chunk["ns"]),
            "1p": len(first_chunk["1p"]),
            "1m": len(first_chunk["1m"]),
            "2p": len(first_chunk["2p"]),
            "2m": len(first_chunk["2m"]),
            }
        columns = list(first_chunk["ns"].keys())
        splitter = MetaDetectSplitter(g, columns, variants)
        for variant in ["ns", "1p", "1m", "2p", "2m"]:
            splitter.write_bin(first_chunk[variant], variant)
        return f

    def write_output(self, outfile, group, data):
        g = outfile[group]
        for variant in ["ns", "1p", "1m", "2p", "2m"]:
            k = g[variant]
            for name, col in data[variant].items():
                # replace masked values with nans
                if np.ma.isMaskedArray(col):
                    col = col.filled(np.nan)
                k[name].append(col) #NOT SURE THIS WORKS EITHER TBD

def sanitize(data):
    """
    Convert unicode arrays into types that h5py can save
    """
    # convert unicode to strings
    if data.dtype.kind == "U":
        data = data.astype("S")
    # convert dates to integers
    elif data.dtype.kind == "M":
        data = data.astype(int)

    return data

# Outstanding issues! 
# - 1 we don't have a fixed length on the things we add to the seperate variants, hence try append? need to figure out if it works
# - 2 same issue means the h5py shorten thing probably wont work? do we need it to or should we just drop it?
# - 3 write the version where data is taken from a parquet file instead of a butler file!
