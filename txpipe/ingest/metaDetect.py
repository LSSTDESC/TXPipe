from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection, MapsFile, TextFile, PNGFile, DataFile
from .lsst import process_metadetect_data, sanitize, process_photometry_data
from .dp_info import DP1_COSMOLOGY_TRACTS, ALL_TRACTS, DP1_TRACTS, TXPIPE_COLUMNS
from ceci.config import StageParameter
from ..utils.hdf_tools import h5py_shorten, repack
from ..utils.splitters import MetaDetectSplitter
from ..shear_calibration.names import META_VARIANTS
import numpy as np
import os
import pyarrow.parquet as pq
import sys
import glob

class TXDP2Ingestion(PipelineStage):
    """
    Base class for things that ingest DP2 using a butler. Do not use directly.
    """
    name = "TXDP2Ingestion"
    inputs = [
        ("tract_list", TextFile)
    ]
    config_options= {
        "butler_repository_index": StageParameter(
            str,
            "/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml",
            msg="Path to repository index"
        ),
        "butler_config_file": StageParameter(
            str, 
            "dp2",
            msg="Path to the LSST butler config file, or code name for it in the repo index"
        ),
        "collections": StageParameter(str, "dp2", msg="Butler collections to use."),
    }

    def get_butler(self):
        os.environ["DAF_BUTLER_REPOSITORY_INDEX"] = self.config["butler_repository_index"]
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

        butler_config_file = self.config["butler_config_file"]
        collections = self.config["collections"]
        try:
            butler = Butler(butler_config_file, collections=collections)
        except Exception as e:
            raise RuntimeError(error_msg) from e

        return butler

    def filter_refs_by_tract(self, refs):
        tracts_file = self.get_input('tract_list')
        if tracts_file == "none":
            print("Not using a tracts file; selecting all objects")
            return refs

        # otherwise, filter on the tract ID
        tracts = np.loadtxt(tracts_file, dtype=int)
        filtered_refs = [ref for ref in refs if ref.dataId["tract"] in tracts]
        return filtered_refs

    def get_maximum_catalog_size(self, butler, refs):
        from pyarrow.parquet import ParquetFile
        n = 0
        nref = len(refs)
        print("Calculating maximum possible catalog size")
        for i, ref in enumerate(refs):
            print(f"Getting file size {i+1} / {nref}")
            uri = butler.getURI('object_shear_all', dataId=ref.dataId)
            p = ParquetFile(uri.ospath)
            n += p.metadata.num_rows         

        # We want a maximum size for each of the sub-catalogs.
        # There are five, and all are close to the same size.
        # In theory one could be a little larger so we give it
        # some wiggle room. This is often not needed as we are
        # usually cutting down by flags anyway but doesn't hurt.
        n = int(n / 5 * 1.05)
        print("Using max row count", n)
        return n


# export DAF_BUTLER_REPOSITORY_INDEX=        

class TXIngestRubinMetaDetect(TXDP2Ingestion):
    """
    Initial ingestion of the Rubin MetaDetect catalog
    """

    name = "TXIngestRubinMetaDetect"
    inputs = [
        ("tract_list", TextFile)
    ]
    outputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options= TXDP2Ingestion.config_options | {
        "exclusion_flag": StageParameter(bool, False, msg="Decide if flags are used for exclusion or just flagged."),
        "flag_list": StageParameter(list, ["is_primary"], msg="list of flags to use for combined."),
        "all_columns": StageParameter(bool, False, msg="do we want to save all columns or just the ones TXPipe needs."),
        "pre_response_shape_noise": StageParameter(float, 0.22, msg="Estimate of shape noise before response for constructing weight.")
    }

    def run(self):

        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"


        butler = self.get_butler()
        data_set_refs = butler.query_datasets('object_shear_all')

        # These options control exactly what we ingest
        all_columns_flag = self.config["all_columns"]
        exclusion_flag = self.config["exclusion_flag"]
        flag_list = self.config["flag_list"]

        used_tract_refs = self.filter_refs_by_tract(data_set_refs)
        n_used_tracts = len(used_tract_refs)
        print(f"Processing {n_used_tracts} tracts")

        shape_noise = self.config['pre_response_shape_noise']

        max_size = self.get_maximum_catalog_size(butler, used_tract_refs)
        created_files = False
        for i, ref in enumerate(used_tract_refs):
            print(f"Processing tract {i + 1} / {n_used_tracts}")
            sys.stdout.flush()
            d = butler.get('object_shear_all',
                           dataId=ref.dataId,
                           )
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"  - skipping chunk since it is empty")
                continue
            else:
                print(f"  - adding {chunk_size} rows")

            shear_data = process_metadetect_data(d, flag_list, exclusion_flag, shape_noise,
                                                 full_columns=all_columns_flag)
            if not created_files:
                created_files = True
                variants = {
                    "ns": max_size,
                    "1p": max_size,
                    "1m": max_size,
                    "2p": max_size,
                    "2m": max_size,
                    }
                columns = list(shear_data["ns"].keys())
                dtypes = {key: shear_data["ns"][key].dtype for key in shear_data["ns"]}
                splitter = MetaDetectSplitter(group, columns, variants, dtypes=dtypes)

            for variant in META_VARIANTS:
                splitter.write_bin(shear_data[variant], variant)
        print("Read complete; re-sizing files")
        if created_files:    
            splitter.finish()
            print("adding in aliases")
            self.aliasing(shear_outfile, group)
        else:
            print("No metadetect data written; skipping splitter.finish/aliasing")
        shear_outfile.close()

        # Repack the files, speeding up future access.
        # This takes a while!
        print("Repacking files")
        repack(self.get_output("shear_catalog"))
    

    def aliasing(self, outfile, group):
        g = group
        for variant in ["ns", "1p", "1m", "2p", "2m"]:
            k = g[variant]
            for txname, original in TXPIPE_COLUMNS.items():
                k[txname] = k[original]


class TXGenerateTractList(TXDP2Ingestion):
    """
    Generate a list of Butler tracts based on a mask.
    """
    name = "TXGenerateTractList"
    inputs = [
    ]
    outputs = [
        ("tract_list", TextFile),
        ("tract_list_plot", PNGFile),
    ]
    config_options = TXDP2Ingestion.config_options | {
        "nside_low": StageParameter(int, 512, msg="The nside resolution for finding tracts from "),
        "dec_min": StageParameter(float, -40.0, msg="Minimum declination to keep. Designed to cut out a little island from a deep field that snuck through"),
    }
    def run(self):
        import healsparse
        import healpy
        from lsst.daf.butler import Butler
        nside_low = self.config['nside_low']
        npix = healpy.nside2npix(nside_low)
        butler = self.get_butler()
        skymap = butler.get("skyMap")

        # This input map is not currently a TXPipe maps file,
        # it is a raw healsparse file, so we don't use open_input,
        # just get the filename
        mask_file_path = self.get_input("shear_mask")
        shear_mask = healsparse.HealSparseMap.read(mask_file_path)
        
        # We make a map at low resolution and see what pixels hit it.
        # I think there is or should be a better way than this. Possibly
        # a newer healsparse version than the one in desc-stack makes this
        # much more straightforward, but using degrade gave nonsensical results
        # here.
        low_res_mask = np.zeros(npix, dtype=bool)

        # Loop through the large coverage pixels. I tried just looking at the
        # coverage map directly, but it looked nothing like that actual high-res
        # mask - lots of empty pixels were included. So instead we need to
        # check in each coverage pixel if there are actually hit pixels there.
        cov_pixels, = np.where(shear_mask._cov_map.coverage_mask)
        n_cov_pix = len(cov_pixels)
        # Loop through the top-level coverage pixels
        for i, cov_pix in enumerate(cov_pixels):
            # get valid pixels in that large pixel
            print(f"Searching pixel {i+1}/{n_cov_pix}")
            d = shear_mask.valid_pixels_single_covpix(cov_pix)
            if d.size == 0:
                continue
            # cut down to True pixels. That's actually all of them in the current version,
            # but let's not rely on that.
            d = d[shear_mask[d]]
            # convert to our resired nside from the high-res sparse maps
            theta, phi = healpy.pix2ang(ipix=d, nside=shear_mask.nside_sparse, nest=True)
            low_pix = healpy.ang2pix(nside_low, theta, phi, nest=True)
            # mark in the medium-res map that this is selected.
            low_res_mask[low_pix] = True

        tracts = set()
        hit_pix = np.where(low_res_mask)[0]
        ra_all, dec_all = healpy.pix2ang(nside_low, hit_pix, nest=True, lonlat=True)
        dec_min = self.config['dec_min']
        cut = dec_all > dec_min
        dec_all = dec_all[cut]
        ra_all = ra_all[cut]
        tracts = np.unique(skymap.findTractIdArray(ra_all, dec_all, degrees=True))
    

        with self.open_output("tract_list") as f:
            np.savetxt(f, tracts, fmt='%i')

        with self.open_output("tract_list_plot", figsize=(8,6), wrapper=True) as fig:
            healpy.mollview(low_res_mask, nest=True, fig=fig.file)
            for i, t in enumerate(tracts):
                plot_tract(skymap, t)

def get_vertices(skymap, tract_id):
    ti = skymap.generateTract(tract_id)
    vl = ti.getVertexList()
    lons = []
    lats = []
    for v in vl:
        ra = v.getLongitude().asDegrees()
        dec = v.getLatitude().asDegrees()
        lons.append(ra)
        lats.append(dec)
    return lons, lats

def plot_tract(skymap, tract_id):
    import healpy
    lons, lats = get_vertices(skymap, tract_id)
    healpy.projplot(lons, lats, 'r-', lonlat=True, linewidth=1)


class TXIngestHealsparseMask(PipelineStage):
    name = "TXIngestHealsparseMask"
    inputs = [
        ("shear_mask", DataFile)
    ]
    outputs = [
        ("mask", MapsFile)
    ]
    config_options = {}

    def run(self):
        import healsparse

        original_path = self.get_input("shear_mask")
        mask = healsparse.HealSparseMap.read(original_path)
        metadata = {
            "pixelization": "healpix",
            "nside": mask.nside_sparse,
            "nest": True,
        }
        with self.open_output("mask", wrapper=True) as f:
            f.write_map("mask", mask, metadata)

class TXIngestDP2Photometry(TXDP2Ingestion):
    name = "TXIngestDP2Photometry"

    inputs = [
        ("tract_list", TextFile)
    ]

    outputs = [
        ("photometry_catalog", PhotometryCatalog)
    ]

    config_options = TXDP2Ingestion.config_options | {
    }



    def run(self):
        from ..utils.hdf_tools import h5py_shorten, repack

        butler = self.get_butler()
        tracts = []

        columns = [
            "objectId",
            "tract",
            "patch",
            "coord_dec",
            "coord_ra",
            "g_cModelFlux",
            "g_cModelFluxErr",
            "g_cModel_flag",
            "i_cModelFlux",
            "i_cModelFluxErr",
            "i_cModel_flag",
            "i_ixx",
            "i_ixxPSF",
            "i_ixy",
            "i_ixyPSF",
            "i_iyy",
            "i_iyyPSF",
            "r_cModelFlux",
            "r_cModelFluxErr",
            "r_cModel_flag",
            "refExtendedness",
            "u_cModelFlux",
            "u_cModelFluxErr",
            "u_cModel_flag",
            "y_cModelFlux",
            "y_cModelFluxErr",
            "y_cModel_flag",
            "z_cModelFlux",
            "z_cModelFluxErr",
            "z_cModel_flag",
            "coord_flag",
            "g_i_flag",
            "r_i_flag",
            "i_i_flag",
            "z_i_flag",
        ]
        data_set_refs = butler.query_datasets("object")
        data_set_refs = self.filter_refs_by_tract(data_set_refs)
        max_cat_size = self.get_maximum_catalog_size(butler, data_set_refs)
        n_chunks = len(data_set_refs)


        created_files = False
        start = 0
        for i, ref in enumerate(data_set_refs):
            d = butler.get("object", dataId=ref.dataId, parameters={"columns": columns})
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"Skipping chunk {i + 1} / {n_chunks} since it is empty")
                continue

            # This renames columns, and does some selection and
            # processing like fluxes to magnitudes and shear moments
            # to shear components.
            data = process_photometry_data(d)

            # If this is the first chunk, we need to create the output files.
            # We only create these here so that if we change the process_photometry_data
            # or process_shear_data methods, we don't have to update the output file creation.
            if not created_files:
                created_files = True
                outfile = self.setup_output(data, max_cat_size)

            # Output these chunks to the output files
            end = start + len(data["ra"])
            self.write_output(outfile, data, start, end)

            print(f"Processing chunk {i + 1} / {n_chunks} into rows {start:,} - {end:,}")
            start = end

        print(f"Final selected objects: {end:,} in photometry")

        # When we created the files we used the maximum possible length
        # for the column sizes (which is what we would get if there were
        # no stars in the catalog or flagged objects). Now we can trim the columns to the
        # actual size of the data we have. Everything after that is empty.
        print("Trimming columns:")
        for col in data.keys():
            print("    ", col)
            h5py_shorten(outfile["photometry"], col, end)

        outfile.close()

        # Run h5repack on the file. This tends to make future access much faster.
        print("Repacking files")
        repack(self.get_output("photometry_catalog"))

    def setup_output(self, first_chunk, n):
        tag = "photometry_catalog"
        group = "photometry"
        f = self.open_output(tag)
        g = f.create_group(group)

        for name, col in first_chunk.items():
            g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f

    def write_output(self, outfile, data, start, end):
        g = outfile["photometry"]
        for name, col in data.items():
            # replace masked values with nans
            if np.ma.isMaskedArray(col):
                col = col.filled(np.nan)
            g[name][start:end] = col




class TXIngestMetaDetectV1_1(PipelineStage):
    """
    Initial ingestion of the Rubin MetaDetect catalog
    """

    name = "TXIngestMetaDetectV1_1"
    inputs = [
    ]
    outputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options= {
        "base_dir": StageParameter(str, "/pscratch/sd/e/esheldon/lsst-mdet-runs/run-dp2-v01.1", msg='Top directory for catalog'),
        "exclusion_flag": StageParameter(bool, False, msg="Decide if flags are used for exclusion or just flagged."),
        "all_columns": StageParameter(bool, False, msg="do we want to save all columns or just the ones TXPipe needs."),
        "pre_response_shape_noise": StageParameter(float, 0.22, msg="Estimate of shape noise before response for constructing weight.")
    }

    def generate_input_file_list(self):
        base_dir = self.config["base_dir"]
        return glob.glob(f"{base_dir}/*/*-mdet.fits")


    def get_maximum_catalog_size(self, file_list):
        import rustfits
        n = 0
        nfile = len(file_list)
        for i, filename in enumerate(file_list):
            if i  and ((i % 10) == 0):
                print(f"Counting rows in file {i} / {nfile}")
            f = rustfits.FITS(filename)
            n += f['cat'].nrows
        print(f"Max row count {n:,}")
        return n


    def run(self):

        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"


        # These options control exactly what we ingest
        all_columns_flag = self.config["all_columns"]
        exclusion_flag = self.config["exclusion_flag"]

        file_list = self.generate_input_file_list()
        n_files = len(file_list)
        shape_noise = self.config['pre_response_shape_noise']

        max_size = self.get_maximum_catalog_size(file_list)
        created_files = False
        for i, filename in enumerate(file_list):
            print(f"Processing tract {i + 1} / {n_files}")
            sys.stdout.flush()
            d = butler.get('object_shear_all',
                           dataId=ref.dataId,
                           )
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"  - skipping chunk since it is empty")
                continue
            else:
                print(f"  - adding {chunk_size} rows")

            shear_data = process_metadetect_data_v1_1(d, exclusion_flag, shape_noise,
                                                 full_columns=all_columns_flag)
            if not created_files:
                created_files = True
                variants = {
                    "ns": max_size,
                    "1p": max_size,
                    "1m": max_size,
                    "2p": max_size,
                    "2m": max_size,
                    }
                columns = list(shear_data["ns"].keys())
                dtypes = {key: shear_data["ns"][key].dtype for key in shear_data["ns"]}
                splitter = MetaDetectSplitter(group, columns, variants, dtypes=dtypes)

            for variant in META_VARIANTS:
                splitter.write_bin(shear_data[variant], variant)
        print("Read complete; re-sizing files")
        if created_files:    
            splitter.finish()
            print("adding in aliases")
            self.aliasing(shear_outfile, group)
        else:
            print("No metadetect data written; skipping splitter.finish/aliasing")
        shear_outfile.close()

        # Repack the files, speeding up future access.
        # This takes a while!
        print("Repacking files")
        repack(self.get_output("shear_catalog"))
    

    def aliasing(self, outfile, group):
        g = group
        for variant in ["ns", "1p", "1m", "2p", "2m"]:
            k = g[variant]
            for txname, original in ERIN_TXPIPE_COLUMNS.items():
                k[txname] = k[original]


ERIN_TXPIPE_COLUMNS = {
    "g1": "g1",
    "g2": "g2",
    "g_cross": "g1g2_cov",
    "T": "T",
    "s2n": "s2n",
    "psf_g1_original": "psfrec_g1",
    "psf_g2_original": "psfrec_g2",
    "psf_T_mean_original": "psfrec_T",
    # The re-convolved psf_g1 is always basically zero.
    # Let's see if we can get away without including it.
    # "psf_g1": "gauss_psfReconvolved_g1",
    # "psf_g2": "gauss_psfReconvolved_g2",
    "psf_T_mean": "psf_T",
    "object_mask_fraction": "mfrac",
    "id": "shearObjectId",
}
def process_metadetect_data_v1_1(data, flag_exclusion, shape_noise, full_columns=False):
    output = {}
    for variant in META_VARIANTS:
        var_data = data[data["mcal_step"] == variant]
        var_data = sanitize(var_data)

        flags = data["flags"]
        if flag_exclusion:
            keep = flags == 0
            var_data = var_data[keep]
            flags = flags[keep]
        if full_columns:
            var_output = {name: var_data[name] for name in var_data.dtype.names} #just process all columns
            var_output.pop("mcal_step", None)
        else:
            needed = sorted(set(ERIN_TXPIPE_COLUMNS.values()) | {"ra", "dec"})
            var_output = {name: var_data[name] for name in needed}
        # extra columns we are still adding:
        var_output["weight"] = 1 / (2 * shape_noise ** 2 + var_data["gauss_g1_g1_Cov"] + var_data["gauss_g2_g2_Cov"])
        var_output["g1_err"] = var_data["g1_err"]
        var_output["g2_err"] = var_data["g1_errr"]

        for band in "griz": # For DP2, we only expect 4 bands
            f = var_data[f"flux_{band}"]
            f_err = var_data[f"flux_err_{band}"]
            var_output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
            var_output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)
        output[f"{variant}"] = var_output

    return output
