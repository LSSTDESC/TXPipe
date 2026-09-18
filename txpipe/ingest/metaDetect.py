from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection, MapsFile, TextFile, PNGFile, DataFile
from .lsst import process_metadetect_data, sanitize
from .dp_info import DP1_COSMOLOGY_TRACTS, ALL_TRACTS, DP1_TRACTS, TXPIPE_COLUMNS
from ceci.config import StageParameter
from ..utils.hdf_tools import h5py_shorten, repack
from ..utils.splitters import MetaDetectSplitter
from ..shear_calibration.names import META_VARIANTS
import numpy as np
import os
import pyarrow.parquet as pq
import sys

class TXIngestRubinMetaDetect(PipelineStage):
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
    config_options= {
        "butler_config_file": StageParameter(
            str, 
            "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
            msg="Path to the LSST butler config file."
        ),
        "cosmology_tracts_only": StageParameter(bool, True, msg="Use only cosmology tracts."),
        "select_field": StageParameter(str, "", msg="Field to select (overrides cosmology_tracts_only)."),
        "select_tracts": StageParameter(list, [], msg="list of tracts (overrides cosmology_tracts_only, but not select_field)."),
        "collections": StageParameter(str, "LSSTComCam/DP1", msg="Butler collections to use."),
        "exclusion_flag": StageParameter(bool, False, msg="Decide if flags are used for exclusion or just flagged."),
        "flag_list": StageParameter(list, ["is_primary"], msg="list of flags to use for combined."),
        "all_columns": StageParameter(bool, False, msg="do we want to save all columns or just the ones TXPipe needs."),
        "pre_response_shape_noise": StageParameter(float, 0.22, msg="Estimate of shape noise before response for constructing weight.")
        }

    def run(self):
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
        try:
            butler = Butler(butler_config_file, collections=collections)
        except Exception as e:
            raise RuntimeError(error_msg) from e

        tracts_file = self.get_input('tract_list')
        if tracts_file != "none":
            print("Using tracts_file:", tracts_file)
            tracts = np.loadtxt(tracts_file, dtype=int)
        # TODO: Update these DP1 thing to make sense for DP2
        elif self.config["select_field"]:
            tracts = DP1_TRACTS[self.config["select_field"]]
        elif self.config["select_tracts"]:
            tracts = self.config["select_tracts"]
        elif self.config["cosmology_tracts_only"]:
            tracts = DP1_COSMOLOGY_TRACTS
        else:
            tracts = ALL_TRACTS
        print(f"ingesting using the following tracts:{tracts}")

        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"

        created_files = False
        data_set_refs = butler.query_datasets('object_shear_all')
        n_chunks = len(data_set_refs)
        all_columns_flag = self.config["all_columns"]
        exclusion_flag = self.config["exclusion_flag"]
        flag_list = self.config["flag_list"]
        used_tract_refs = [ref for ref in data_set_refs if ref.dataId["tract"] in tracts]
        n_used_tracts = len(used_tract_refs)
        print(f"Processing {n_used_tracts} tracts")

        shape_noise = self.config['pre_response_shape_noise']

        max_size = self.get_maximum_size(butler, used_tract_refs)

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
        # We have temporarily commented out the repack step
        # as it was crazily slow, taking far longer than the original
        # run.
        # print("Repacking files")
        # repack(self.get_output("shear_catalog"))
    
    def get_maximum_size(self, butler, refs):
        from pyarrow.parquet import ParquetFile
        n = 0
        for ref in refs:
            uri = butler.getURI('object_shear_all', dataId=ref.dataId)
            p = ParquetFile(uri.ospath)
            n += p.metadata.num_rows         

        # We want a maximum size for each of the sub-catalogs.
        # There are five, and all are close to the same size.
        # In theory one could be a little larger so we give it
        # some wiggle room. This is often not needed as we are
        # usually cutting down by flags anyway but doesn't hurt.
        return int(n / 5 * 1.05)


    def aliasing(self, outfile, group):
        g = group
        for variant in ["ns", "1p", "1m", "2p", "2m"]:
            k = g[variant]
            for txname, original in TXPIPE_COLUMNS.items():
                k[txname] = k[original]


class TXGenerateTractList(PipelineStage):
    name = "TXGenerateTractList"
    inputs = [
        ("shear_mask", DataFile)
    ]
    outputs = [
        ("tract_list", TextFile),
        ("tract_list_plot", PNGFile),
    ]
    config_options = {
        "nside_low": StageParameter(int, 512, msg="The nside resolution for finding tracts from "),
        "collections": StageParameter(str, "LSSTComCam/DP2", msg="Butler collections to use."),
        "dec_min": StageParameter(float, -40.0, msg="Minimum declination to keep. Designed to cut out a little island from a deep field that snuck through"),
        "butler_config_file": StageParameter(
            str, 
            "/global/cfs/cdirs/lsst/production/gen3/rubin/DP2/repo/butler.yaml",
            msg="Path to the LSST butler config file."
        ),
    }
    def run(self):
        import healsparse
        import healpy
        from lsst.daf.butler import Butler
        nside_low = self.config['nside_low']
        npix = healpy.nside2npix(nside_low)
        butler_config_file = self.config['butler_config_file']
        collections = self.config['collections']
        butler = Butler(butler_config_file, collections=collections)
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
