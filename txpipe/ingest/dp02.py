from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, HDFFile
from ..utils import band_variants, metacal_variants, nanojansky_err_to_mag_ab, nanojansky_to_mag_ab, moments_to_shear, mag_ab_to_nanojansky
import numpy as np
import glob
import re
import math


class TXIngestDataPreview02(PipelineStage):
    """
    Ingest galaxy catalogs from DP0.2

    There is no metacal on this, and there won't be.

    """
    name = "TXIngestDataPreview02"
    parallel = False
    inputs = []

    outputs = [
        ("photometry_catalog", HDFFile),
        ("shear_catalog", HDFFile),
    ]
    config_options = {
        "pq_path": "/global/cfs/cdirs/lsst/shared/rubin/DP0.2/objectTable/",
        "tracts": "",
    }

    def run(self):
        from pyarrow.parquet import ParquetFile
        import h5py
        from ..utils.hdf_tools import h5py_shorten, repack

        tracts = self.config["tracts"]
        pq_path = self.config['pq_path']

        cat_files = glob.glob(f"{pq_path}/objectTable*.parq")

        if tracts:
            tracts = [tract.strip() for tract in tracts.split(',')]
            print(f"Using {len(tracts)} tracts out of {len(cat_files)}")
            cat_files = [c for c in cat_files if c.split("/")[-1].split("_")[2] in tracts]
            if len(cat_files) != len(tracts):
                raise ValueError("Some tracts not found")                

        n = 0
        for fn in cat_files:
            with ParquetFile(fn) as f:
                n += f.metadata.num_rows

        print(f"Full catalog size = {n:,}")


        # Input columns for photometry
        photo_cols = ["objectId", "coord_ra", "coord_dec", "refExtendedness", "tract"]

            
            
        shape_cols = ["objectId", "coord_ra", "coord_dec", "refExtendedness", "tract", 
                      "i_hsmShapeRegauss_e1", "i_hsmShapeRegauss_e2", "i_hsmShapeRegauss_sigma", "i_hsmShapeRegauss_flag",
                      "i_ixx", "i_ixy", "i_iyy",
                      "i_ixxPSF", "i_ixyPSF", "i_iyyPSF",
                     ]

        # Magnitude columns, given to both photometry and shear catalogs
        for band in "ugrizy":
            for cols in [photo_cols, shape_cols]:
                cols.append(f"{band}_cModelFlux")
                cols.append(f"{band}_cModelFluxErr")
                cols.append(f"{band}_cModel_flag")

        cols = list(set(shape_cols + photo_cols))

        nfile = len(cat_files)

        # This is the default batch size for pyarrow
        batch_size = 65536
        start1 = 0
        start2 = 0
        for i, fn in enumerate(cat_files):
            with ParquetFile(fn) as f:
                n_chunk = math.ceil(f.metadata.num_rows / batch_size)
                it = f.iter_batches(columns=cols)
                for j,d in enumerate(it):
                    d = {col.name: d[col.name].to_numpy(zero_copy_only=False) for col in d.schema}
                    if i == 0 and j == 0:
                        output_names = set(d.keys())
                        for col in cols:
                            assert col in output_names, f"Column {col} not found"
                    
                    photo_data = self.process_photometry_data(d)
                    shear_data = self.process_shear_data(d)

                    if i == 0 and j == 0:
                        photo_outfile = self.setup_output("photometry_catalog", "photometry", photo_data, n)
                        shear_outfile = self.setup_output("shear_catalog", "shear", shear_data, n)

                    end1 = start1 + len(photo_data['ra'])
                    end2 = start2 + len(shear_data['ra'])
                    self.write_output(photo_outfile, "photometry", photo_data, start1, end1)
                    self.write_output(shear_outfile, "shear", shear_data, start2, end2)
                    print(f"Processing chunk {j+1}/{n_chunk} of file {i+1}/{nfile} into rows {start1:,} - {end1:,}")
                    start1 = end1
                    start2 = end2


        print("Final selected objects: {end1:,} in photometry and {end2:,} in shear")
        # Cut down to just include stars.
        print("Trimming photometry columns:")
        for col in photo_data.keys():
            print("    ", col)
            h5py_shorten(photo_outfile["photometry"], col, end1)

        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, end2)

        photo_outfile.close()
        shear_outfile.close()

        # Run h5repack on the file
        print("Repacking file")
        repack(self.get_output("photometry_catalog"))
        repack(self.get_output("shear_catalog"))



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
        output["m"] = np.repeat(-1.184445e-01, f.size)
        output["c1"] = np.repeat(-2.316957e-04, f.size)
        output["c2"] = np.repeat(-8.629799e-05, f.size)
        output["sigma_e"] = np.repeat(1.342084e-01, f.size)
        output["weight"] = np.ones_like(f)

        # PSF components
        output["psf_T_mean"] = data['i_ixxPSF'][cut] + data['i_iyyPSF'][cut]
        psf_g1, psf_g2 = moments_to_shear(data['i_ixxPSF'][cut], data['i_iyyPSF'][cut], data['i_ixyPSF'][cut])
        output["psf_g1"] = psf_g1
        output["psf_g2"] = psf_g2

            
            
        return output

