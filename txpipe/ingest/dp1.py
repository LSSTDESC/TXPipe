from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile
from ..utils import nanojansky_err_to_mag_ab, nanojansky_to_mag_ab, moments_to_shear, mag_ab_to_nanojansky
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
        butler_config_file = self.config["butler_config_file"]
        butler = Butler(butler_config_file, collections="LSSTComCam/DP1")
        self.ingest_photometry(butler)


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
            photo_data = self.process_photometry_data(d)
            shear_data = self.process_shear_data(d)

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
        pass




    def setup_output(self, tag, group, first_chunk, n):
        import h5py

        f = self.open_output(tag)
        g = f.create_group(group)

        for name, col in first_chunk.items():
            g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f

    def write_output(self, outfile, group, data, start, end):
        g = outfile[group]
        for name, col in data.items():
            g[name][start:end] = col

    def process_photometry_data(self, data):
        cut = data['refExtendedness'] == 1
        cols = {
            'ra': 'coord_ra',
            'dec': 'coord_dec',
            'tract': 'tract',
            'id': 'objectId',
            'extendedness': 'refExtendedness'
        }
        output = {new_name: data[old_name][cut] for new_name, old_name in cols.items()}
        for band in "ugrizy":
            f = data[f"{band}_cModelFlux"][cut]
            f_err = data[f"{band}_cModelFluxErr"][cut]
            output[f'mag_{band}'] = nanojansky_to_mag_ab(f)
            output[f'mag_err_{band}'] = nanojansky_err_to_mag_ab(f, f_err)
            output[f'snr_{band}'] = f / f_err

            # for undetected objects we use a mock mag of 30
            # to choose mag errors
            f_mock = mag_ab_to_nanojansky(30.0)
            undetected = f <= 0
            output[f'mag_{band}'][undetected] = np.inf
            output[f'mag_err_{band}'][undetected] = nanojansky_err_to_mag_ab(f_mock, f_err[undetected])
            output[f'snr_{band}'][undetected] = 0.0
        return output

    def process_shear_data(self, data):
        cut = data['refExtendedness'] == 1
        cols = {
            'ra': 'coord_ra',
            'dec': 'coord_dec',
            'tract': 'tract',
            'id': 'objectId',
            'extendedness': 'refExtendedness'
        }
        output = {new_name: data[old_name][cut] for new_name, old_name in cols.items()}
        for band in "ugrizy":
            f = data[f"{band}_cModelFlux"][cut]
            f_err = data[f"{band}_cModelFluxErr"][cut]
            output[f'mag_{band}'] = nanojansky_to_mag_ab(f)
            output[f'mag_err_{band}'] = nanojansky_err_to_mag_ab(f, f_err)

            if band == "i":
                output['s2n'] = f / f_err
        
        output["g1"] = data['i_hsmShapeRegauss_e1'][cut]
        output["g2"] = data['i_hsmShapeRegauss_e2'][cut]
        output["T"] = data['i_ixx'][cut] + data['i_iyy'][cut]
        output["flags"] = data["i_hsmShapeRegauss_flag"][cut]


        # Fake numbers! These need to be derived from simulation.
        # In this case 
        # output["m"] = np.repeat(0.0, f.size)
        # output["c1"] = np.repeat(-2.316957e-04, f.size)
        # output["c2"] = np.repeat(-8.629799e-05, f.size)
        # output["sigma_e"] = np.repeat(1.342084e-01, f.size)
        output["weight"] = np.ones_like(f)

        # PSF components
        output["psf_T_mean"] = data['i_ixxPSF'][cut] + data['i_iyyPSF'][cut]
        psf_g1, psf_g2 = moments_to_shear(data['i_ixxPSF'][cut], data['i_iyyPSF'][cut], data['i_ixyPSF'][cut])
        output["psf_g1"] = psf_g1
        output["psf_g2"] = psf_g2

            
            
        return output




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
