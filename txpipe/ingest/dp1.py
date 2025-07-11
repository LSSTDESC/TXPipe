from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile
from .lsst import process_photometry_data, process_shear_data
import numpy as np

class TXIngestDataPreview1(PipelineStage):
    """
    Ingest galaxy catalogs from DP1
    """
    name = "TXIngestDataPreview1"
    inputs = []
    outputs = [
        ("photometry_catalog", PhotometryCatalog),
        ("shear_catalog", ShearCatalog),
        ("exposures", HDFFile),
    ]
    config_options = {
        "butler_config_file": "/global/cfs/cdirs/lsst/production/gen3/rubin/dp1/repo/butler.yaml",
    }

    def run(self):
        from lsst.daf.butler import Butler

        # Configure and create the butler. There seem to be several ways
        # to do this, and there is a central collective butler yaml file
        # on NERSC
        butler_config_file = self.config["butler_config_file"]
        butler = Butler(butler_config_file, collections="LSSTComCam/DP1")

        self.ingest_photometry(butler)
        self.ingest_visits(butler)


    def ingest_photometry(self, butler):
        from ..utils.hdf_tools import h5py_shorten, repack
        columns = [
            'objectId',
            'tract',
            'coord_dec',
            'coord_ra',
            'g_cModelFlux',
            'g_cModelFluxErr',
            'g_cModel_flag',
            'i_cModelFlux',
            'i_cModelFluxErr',
            'i_cModel_flag',
            'i_hsmShapeRegauss_e1',
            'i_hsmShapeRegauss_e2',
            'i_hsmShapeRegauss_flag',
            'i_hsmShapeRegauss_sigma',
            'i_ixx',
            'i_ixxPSF',
            'i_ixy',
            'i_ixyPSF',
            'i_iyy',
            'i_iyyPSF',
            'r_cModelFlux',
            'r_cModelFluxErr',
            'r_cModel_flag',
            'refExtendedness',
            'u_cModelFlux',
            'u_cModelFluxErr',
            'u_cModel_flag',
            'y_cModelFlux',
            'y_cModelFluxErr',
            'y_cModel_flag',
            'z_cModelFlux',
            'z_cModelFluxErr',
            'z_cModel_flag'
        ]

        n = self.get_catalog_size(butler, "object")

        created_files = False
        photo_start = 0
        shear_start = 0
        data_set_refs = butler.query_datasets("object")
        n_chunks = len(data_set_refs)
        for i, ref in enumerate(data_set_refs):
            d = butler.get("object", dataId=ref.dataId, parameters={'columns': columns})
            chunk_size = len(d)

            if chunk_size == 0:
                continue

            # This renames columns, and does some selection and
            # processing like fluxes to magnitudes and shear moments
            # to shear components.
            photo_data = process_photometry_data(d)
            shear_data = process_shear_data(d)

            # If this is the first chunk, we need to create the output files.
            # We only create these here so that if we change the process_photometry_data
            # or process_shear_data methods, we don't have to update the output file creation.
            if not created_files:
                created_files = True
                photo_outfile = self.setup_output("photometry_catalog", "photometry", photo_data, n)
                shear_outfile = self.setup_output("shear_catalog", "shear", shear_data, n)
                # We don't have a good shear catalog yet, so all our shears are going to be
                # uncalibrated. So let's not even try to calibrate them, and instead just 
                # pretend they are precalibrated.
                shear_outfile["shear"].attrs["catalog_type"] = "precalibrated"

            # Output these chunks to the output files
            photo_end = photo_start + len(photo_data['ra'])
            shear_end = shear_start + len(shear_data['ra'])
            self.write_output(photo_outfile, "photometry", photo_data, photo_start, photo_end)
            self.write_output(shear_outfile, "shear", shear_data, shear_start, shear_end)

            print(f"Processing chunk {i+1} / {n_chunks} into rows {photo_start:,} - {photo_end:,}")
            photo_start = photo_end
            shear_start = shear_end

        print("Final selected objects: {photo_end:,} in photometry and {shear_end:,} in shear")

        # When we created the files we used the maximum possible length
        # for the column sizes (which is what we would get if there were
        # no stars in the catalog). Now we can trim the columns to the
        # actual size of the data we have. Everything after that is empty.
        print("Trimming columns:")
        for col in photo_data.keys():
            print("    ", col)
            h5py_shorten(photo_outfile["photometry"], col, photo_end)
        
        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, shear_end)

        photo_outfile.close()
        shear_outfile.close()

        # Run h5repack on the file
        print("Repacking files")
        repack(self.get_output("photometry_catalog"))
        repack(self.get_output("shear_catalog"))

    def ingest_visits(self, butler):

        # There aren't that many columns, we can just dump the whole thing
        with self.open_output("exposures") as f:
            d1 = butler.get("visit_table")
            g = f.create_group("visits")
            for col in d1.columns:
                data = sanitize(d1[col])
                g.create_dataset(col, data=data)

            # Let's also save the detector visits table as we can use
            # if for null tests on chip center tangential shear.
            g = f.create_group("detector_visits")
            d2 = butler.get("visit_detector_table")
            for col in d2.columns:
                data = sanitize(d2[col])
                g.create_dataset(col, data=data)



    def setup_output(self, tag, group, first_chunk, n):
        f = self.open_output(tag)
        g = f.create_group(group)

        for name, col in first_chunk.items():
            g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f

    def write_output(self, outfile, group, data, start, end):
        g = outfile[group]
        for name, col in data.items():
            g[name][start:end] = col



    def get_catalog_size(self, butler, dataset_type):
        import pyarrow.parquet
        n = 0
        for ref in butler.query_datasets(dataset_type):
            uri = butler.getURI(ref)
            if not uri.path.endswith(".parq"):
                raise ValueError(f"Some data in dataset {dataset_type} was not in parquet format: {uri.path}")
            with pyarrow.parquet.ParquetFile(uri.path) as f:
                n += f.metadata.num_rows
        return n


def sanitize(data):
    """
    Convert unicode arrays into types that h5py can save
    """
    # convert unicode to strings
    if data.dtype.kind == "U":
        data = data.astype("S")
    # convert dates to integers
    elif data.dtype.kind == "M":
        data = data.astype(int)

    return data
