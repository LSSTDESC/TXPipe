from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, HDFFile
from ..utils import band_variants, metacal_variants, nanojansky_err_to_mag_ab, nanojansky_to_mag_ab
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
        s1 = 0
        s2 = 0
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

                    e1 = s1 + len(photo_data['ra'])
                    e2 = s2 + len(shear_data['ra'])
                    self.write_output(photo_outfile, "photometry", photo_data, s1, e1)
                    self.write_output(shear_outfile, "shear", shear_data, s2, e2)
                    print(f"Processing chunk {j+1}/{n_chunk} of file {i+1}/{nfile} into rows {s1:,} - {e1:,}")
                    s1 = e1
                    s2 = e2


        print("Final selected objects: {e1:,} in photometry and {e2:,} in shear")
        # Cut down to just include stars.
        print("Trimming photometry columns:")
        for col in photo_data.keys():
            print("    ", col)
            h5py_shorten(photo_outfile["photometry"], col, e1)

        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, e2)

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
        
        output["g1"] = data['i_hsmShapeRegauss_e1'][cut]
        output["g2"] = data['i_hsmShapeRegauss_e2'][cut]
        output["T"] = data['i_ixx'][cut] + data['i_iyy'][cut]
        output["flags"] = data["i_hsmShapeRegauss_flag"][cut]
        output['s2n'] = f / f_err

        # Fake numbers! These need to be derived from simulation.
        # In this case 
        output["m"] = np.repeat(-1.184445e-01, f.size)
        output["c1"] = np.repeat(-2.316957e-04, f.size)
        output["c2"] = np.repeat(-8.629799e-05, f.size)

        output["sigma_e"] = np.repeat(1.342084e-01, f.size)
        output["weight"] = np.ones_like(f)
        output["psf_T_mean"] = data['i_ixxPSF'][cut] + data['i_iyyPSF'][cut]

            
            
        return output

