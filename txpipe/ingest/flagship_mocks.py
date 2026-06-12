import glob
import numpy as np
from ..base_stage import PipelineStage
from ..data_types import ShearCatalog
from ..utils.conversion import mag_ab_to_nanojansky_with_errors, combine_intrinsic_shear
from ..utils.hdf_tools import repack
from ceci.config import StageParameter

DEFAULT_MOCK_PATTERN = "/global/cfs/cdirs/lsst/groups/PZ/users/qhang/flagship_dp2/flagship_mock/Flagship/dp2_mock_run_flagship_gold_test/248*/output_deredden_lsst_obs_cond_dp2.pq"

class TXIngestFlagshipMocks(PipelineStage):
    name = "TXIngestFlagshipMocks"
    inputs = []
    outputs = [
        ("shear_catalog", ShearCatalog)
    ]
    config_options = {
        "input_file_pattern": StageParameter(str, default=DEFAULT_MOCK_PATTERN, msg="Glob pattern for input files"),
    }

    def run(self):
        
        input_files = glob.glob(self.config["input_file_pattern"])
        if not input_files:
            raise ValueError("No files found with pattern: ", self.config["input_file_pattern"])
        
        # This is a maximum size - actually we will be quite a bit
        # smaller that this. We will cut the catalog down at the end.
        size = self.get_catalog_size(input_files)

        # Set up the output file objects
        filename = self.get_output("shear_catalog")
        outfile, outgroup = setup_mock_shear_catalog_file(filename, size)

        # Loop through the chunks of data in the different input files
        s = 0
        for data in self.iterate_data(input_files):

            # Convert the data in each to the observable
            # quantities we might see in a real catalog
            data = self.process_chunk(data)
            e = s + len(data["ra"])

            # save this chunk to the file
            print(f"Saving rows {s} - {e}")
            for name, col in data.keys():
                outgroup[name][s:e] = col
            s = e
        
        # reclaim the space because we over-estimated the availeble size
        for key in list(outgroup.keys()):
            outgroup[key].resize(s)
        
        outfile.close()

        # This re-packs everything to save space
        # because we did the resizing
        print("Repacking")
        repack(filename)

        

    def get_catalog_size(self, input_files):
        from pyarrow.parquet import ParquetFile
        size = 0
        for filename in input_files:
            with ParquetFile(filename) as f:
                size += f.metadata.num_rows
        return size
    
    def process_chunk(self, chunk):
        n = len(chunk["ra"])
        data = {}

        # First do the columns which are copied directly
        # unchanged from the input
        as_is_columns = ["ra", "dec"]
        for col in as_is_columns:
            data[col] = chunk[col]

        # next the ones that are just renamed values
        renamed_columns = {
            "galaxy_id": "id",
            "redshift": "redshift_true",
            "true_g1": "gamma1",
            "true_g2": "gamma2",
        }
        for b in "ugrizy":
            renamed_columns[f"mag_{b}_lsst"] = f"mag_{b}"
            renamed_columns[f"mag_{b}_lsst_err"] = f"mag_err_{b}"

        for old_name, new_name in renamed_columns.items():
            data[new_name] = chunk[old_name]

        # Now the ones we actually need to derive
        T = half_light_radius_to_T(chunk['totalHalfLightRadiusArcsec'])
        data["s2n"] = mags_to_snr(data, "gri")
        data["g1"], data["g2"] = combine_intrinsic_shear(chunk["gamma1"], chunk["gamma2"], chunk["eps1_gal"], chunk["eps2_gal"])
        data["weight"] = wildly_simplistic_weight_model(data)

        # and finally some mock columns that matter for diagnostics
        # but we can leave to zero here. This will break the diagnostics stage!
        data["flags"] = np.zeros(n, dtype=np.int64)
        data["psf_T_mean"] = np.ones(n, dtype=np.float64)
        data["psf_g1"] = np.zeros(n, dtype=np.float64)
        data["psf_g2"] = np.zeros(n, dtype=np.float64)

        # cut down to just the objects with SNR > 4.
        # should do this earlier really, it would be a bit
        # more efficient, but I suspect the IO will dominate.
        cut = (data["s2n"] > 4)
        data = {name: col[cut] for name, col in data.items()}
        return data



    def iterate_data(self, input_files):
        from pyarrow.parquet import ParquetFile
        for filename in input_files:
            with ParquetFile(filename) as f:
                for batch in f.iter_batches():
                    batch = {k:np.array(batch[k]) for k in batch.column_names}
                    yield batch


def half_light_radius_to_T(hlr):
    # HLR -> T assuming gaussian profile.
    # very approximate
    sigma = hlr / np.sqrt(2 * np.log(2))
    T = 2 * sigma**2
    return T

def wildly_simplistic_weight_model(data):
    """
    JZ Threw toegether this very rough fit to the most
    prominent line in the scatter plot of w vs SNR.
    Actually it should depend on size, so I guess this will
    be mainly fit to small galaxies. Do not use for anything
    where weights actually matter!
    """
    snr = data['s2n']
    wmax = 40.0
    return 1.0 / (wmax**-1.0 + 3 * snr**-2.0)



def setup_mock_shear_catalog_file(filename, size):
    import h5py
    f = h5py.File(filename, "w")
    g = f.create_group("shear")
    g.attrs['catalog_type'] = "simple"
    g.create_dataset("galaxy_id", shape=(size,), maxshape=(size,), dtype=np.int64)
    for col in [
        "T", 
        "T_err", 
        "dec", 
        "flags", 
        "g1", 
        "g2", 
        "psf_T_mean", 
        "psf_g1", 
        "psf_g2", 
        "ra", 
        "redshift_true", 
        "s2n", 
        "weight",
    ]:
        g.create_dataset(col, shape=(size,), maxshape=(size, ), dtype=np.float64)
    return f, g

