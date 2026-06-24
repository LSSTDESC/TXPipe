import glob
import numpy as np
from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, HDFFile
from ..utils.conversion import mag_ab_to_nanojansky_with_errors, combine_intrinsic_shear, mags_to_snr
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
        "chunk_rows": StageParameter(int, default=1_000_000, msg="Number of rows to process at once")
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
        s1 = 0
        for data in self.iterate_data(input_files):
            e1 = s1 + len(data["ra"])
            print(f"Processing rows {s1:,} - {e1:,} of {size:,}")
            s1 = e1

            # Convert the data in each to the observable
            # quantities we might see in a real catalog
            data = self.process_chunk(data)
            # New end value because we throw away some data.
            e = s + len(data["ra"])


            
            # save this chunk to the file
            for name, col in data.items():
                outgroup[name][s:e] = col
            s = e
        
        # reclaim the space because we over-estimated the availeble size
        print(f"Cutting down to final catalog size {size} -> {e}")
        for key in list(outgroup.keys()):
            outgroup[key].resize((s,))
        
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
            "gamma1": "true_g1",
            "gamma2": "true_g2",
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
        batch_size = self.config["chunk_rows"]
        print("Using batch size: ", batch_size)
        for filename in input_files:
            with ParquetFile(filename) as f:
                for batch in f.iter_batches(batch_size):
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
    prominent line in the scatter plot of w vs SNR from DES Y3.
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
    g.create_dataset("id", shape=(size,), maxshape=(size,), dtype=np.int64)
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
        "true_g1",
        "true_g2",    
        "s2n", 
        "weight",
    ]:
        g.create_dataset(col, shape=(size,), maxshape=(size, ), dtype=np.float64)
    for b in "ugrizy":
        g.create_dataset("mag_"+b, shape=(size,), maxshape=(size, ), dtype=np.float64)
        g.create_dataset("mag_err_"+b, shape=(size,), maxshape=(size, ), dtype=np.float64)
    return f, g




COLUMN_MAP = {
    # LSST magnitudes
    "mag_u_lsst": "mag_u",
    "mag_g_lsst": "mag_g",
    "mag_r_lsst": "mag_r",
    "mag_i_lsst": "mag_i",
    "mag_z_lsst": "mag_z",
    "mag_y_lsst": "mag_y",
    # LSST magnitude errors
    "mag_u_lsst_err": "mag_u_err",
    "mag_g_lsst_err": "mag_g_err",
    "mag_r_lsst_err": "mag_r_err",
    "mag_i_lsst_err": "mag_i_err",
    "mag_z_lsst_err": "mag_z_err",
    "mag_y_lsst_err": "mag_y_err",
    # Euclid magnitudes (kept as-is)
    "mag_y_euclid_nisp": "mag_y_euclid_nisp",
    "mag_j_euclid_nisp": "mag_j_euclid_nisp",
    "mag_h_euclid_nisp": "mag_h_euclid_nisp",
    "mag_vis_euclid": "mag_vis_euclid",
    # Astrometry
    "ra": "ra",
    "dec": "dec",
    # Identifiers / pixelisation
    "halo_id": "halo_id",
    "galaxy_id": "galaxy_id",
    "pixel": "pixel",
    # Physical properties
    "log_sfr": "log_sfr",
    "log_stellar_mass": "log_stellar_mass",
    "lm_halo": "lm_halo",
    "redshift": "redshift_true",
    # Morphology
    "totalHalfLightRadiusArcsec": "totalHalfLightRadiusArcsec",
    "major": "major",
    "minor": "minor",
}

# Integer-typed HDF5 columns (everything else is float64)
_INT_PARQUET_COLS = {"halo_id", "galaxy_id", "pixel"}

# Columns that may legitimately be absent from some catalogs (e.g. Cardinal
# has no halo catalogue and no Euclid VIS band).  Missing optional columns are
# written as zeros in the output HDF5.
OPTIONAL_PARQUET_COLUMNS = {"halo_id", "log_sfr", "log_stellar_mass", "lm_halo", "mag_vis_euclid"}

# Parquet columns we actually need to read
PARQUET_COLUMNS = list(COLUMN_MAP.keys())

# After pre-allocating datasets, also create mag_err_* aliases
ERR_ALIASES = {
    "mag_u_lsst_err": "mag_err_u",
    "mag_g_lsst_err": "mag_err_g",
    "mag_r_lsst_err": "mag_err_r",
    "mag_i_lsst_err": "mag_err_i",
    "mag_z_lsst_err": "mag_err_z",
    "mag_y_lsst_err": "mag_err_y",
}




class TXIngestParquetCatalog(PipelineStage):
    """Ingest one or more parquet simulation catalogs into a single TXPipe
    HDF5 photometry catalog.

    Multiple files are concatenated row-wise in the order supplied and
    streamed chunk-by-chunk, so memory usage is bounded by ``chunk_rows``
    rather than by the total catalog size.

    All input files must share the same column schema.  If a column is
    missing in any file an informative error is raised immediately (before
    any data is written).

    Parameters
    ----------
    parquet_paths : str or list[str]
        A single path, a YAML list of paths, or a glob pattern
        (e.g. ``/data/catalog_*.parquet``).  Glob patterns are resolved
        in alphabetical order.
    chunk_rows : int
        Number of rows to read from each file per iteration (default 100 000).
    """

    name = "TXIngestParquetCatalog"
    parallel = False

    inputs = []   # parquet files are given via config (no built-in ceci type)

    outputs = [
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "parquet_paths": StageParameter(
            str,
            msg=(
                "Path(s) to input parquet files. "
                "Accepts a single path, a YAML list of paths, or a glob pattern."
            ),
        ),
        "chunk_rows": StageParameter(
            int, 100_000, msg="Rows to read per chunk per file"
        ),
    }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_paths(self):
        import glob as _glob
        import ast

        raw = self.config["parquet_paths"]

        if isinstance(raw, list):
            # ceci correctly parsed a YAML list
            paths = [str(p) for p in raw]
        elif isinstance(raw, str) and raw.strip().startswith('['):
            # ceci passed the YAML list as a stringified Python list
            # e.g. "['/path/a.pq', '/path/b.pq']"
            paths = [str(p) for p in ast.literal_eval(raw)]
        else:
            # Single path or glob pattern
            expanded = sorted(_glob.glob(raw))
            paths = expanded if expanded else [str(raw)]

        if not paths:
            raise ValueError(
                f"No parquet files found for pattern/path: {raw!r}"
            )

        return paths

    def _count_rows(self, path):
        """Return the row count of *path* without loading any column data."""
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows

    def _validate_columns(self, path):
        """Raise ``KeyError`` listing every missing *required* column in *path*.

        Optional columns (OPTIONAL_PARQUET_COLUMNS) are allowed to be absent;
        they will be filled with zeros in the output.
        """
        import pyarrow.parquet as pq
        schema_names = set(pq.read_schema(path).names)
        required = [c for c in PARQUET_COLUMNS if c not in OPTIONAL_PARQUET_COLUMNS]
        missing_required = [c for c in required if c not in schema_names]
        if missing_required:
            raise KeyError(
                f"File {path!r} is missing required columns: {missing_required}\n"
                f"Available columns in that file: {sorted(schema_names)}"
            )
        missing_optional = [c for c in OPTIONAL_PARQUET_COLUMNS if c not in schema_names]
        if missing_optional:
            print(f"  Optional columns absent (will be zero-filled): {missing_optional}")

    def _iter_file_chunks(self, path):
        """Yield ``(batch_size, pandas.DataFrame)`` for every chunk of *path*.

        Only columns that are actually present in the file are requested from
        pyarrow; missing optional columns are added as zero-filled Series.
        """
        import pyarrow.parquet as pq
        schema_names = set(pq.read_schema(path).names)
        cols_to_read = [c for c in PARQUET_COLUMNS if c in schema_names]
        missing_optional = [c for c in PARQUET_COLUMNS
                            if c not in schema_names and c in OPTIONAL_PARQUET_COLUMNS]
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=self.config["chunk_rows"], columns=cols_to_read):
            df = batch.to_pandas()
            for col in missing_optional:
                df[col] = 0
            yield len(df), df

    def _prepare_output(self, nobj_total, output_file):
        g = output_file.create_group("photometry")
        g.attrs["nobj"] = nobj_total

        for parq_col, hdf_col in COLUMN_MAP.items():
            dtype = np.int64 if parq_col in _INT_PARQUET_COLS else np.float64
            g.create_dataset(hdf_col, shape=(nobj_total,), dtype=dtype)

        # Also create mag_err_* aliases expected by TXLensCatalogSplitter
        for parq_col, hdf_col in ERR_ALIASES.items():
            dtype = np.float64
            g.create_dataset(hdf_col, shape=(nobj_total,), dtype=dtype)

        return g

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        print("DEBUG: run() has been called", flush=True)
        import pyarrow.parquet as pq  # verify available

        paths = self._resolve_paths()
        n_files = len(paths)
        print(f"TXIngestParquetCatalog: ingesting {n_files} file(s).", flush=True)

        # 1. Validate and count rows across all files
        file_row_counts = []
        for i, path in enumerate(paths):
            print(f"  [{i+1}/{n_files}] Checking: {path}", flush=True)
            self._validate_columns(path)
            n = self._count_rows(path)
            file_row_counts.append(n)
            print(f"           {n:,} rows", flush=True)

        nobj_total = sum(file_row_counts)
        print(f"  Grand total: {nobj_total:,} rows.", flush=True)

        # 2. Open output and pre-allocate datasets
        output_file = self.open_output("photometry_catalog")
        phot_group = output_file.create_group("photometry")
        phot_group.attrs["nobj"] = nobj_total

        for parq_col, hdf_col in COLUMN_MAP.items():
            dtype = np.int64 if parq_col in _INT_PARQUET_COLS else np.float64
            phot_group.create_dataset(hdf_col, shape=(nobj_total,), dtype=dtype)

        # Also pre-allocate mag_err_* aliases expected by TXLensCatalogSplitter
        # (your parquet has mag_g_err etc. but TXPipe expects mag_err_g etc.)
        for parq_col, hdf_col in ERR_ALIASES.items():
            phot_group.create_dataset(hdf_col, shape=(nobj_total,), dtype=np.float64)

        # 3. Stream each file into the pre-allocated datasets
        global_row = 0
        for file_idx, (path, file_nrows) in enumerate(zip(paths, file_row_counts)):
            file_end = global_row + file_nrows
            print(f"  [{file_idx+1}/{n_files}] Writing: {path}", flush=True)

            chunk_start = global_row
            for n_batch, df in self._iter_file_chunks(path):
                chunk_end = chunk_start + n_batch

                # Write primary columns (mag_g_err convention)
                for parq_col, hdf_col in COLUMN_MAP.items():
                    phot_group[hdf_col][chunk_start:chunk_end] = df[parq_col].to_numpy()

                # Write mag_err_* aliases (mag_err_g convention)
                for parq_col, hdf_col in ERR_ALIASES.items():
                    phot_group[hdf_col][chunk_start:chunk_end] = df[parq_col].to_numpy()

                print(f"    rows {chunk_start:,} – {chunk_end-1:,}", flush=True)
                chunk_start = chunk_end

            if chunk_start != file_end:
                raise RuntimeError(
                    f"Row count mismatch in {path!r}: "
                    f"expected {file_nrows}, got {chunk_start - global_row}"
                )
            global_row = file_end

        assert global_row == nobj_total
        output_file.flush()
        output_file.close()
        print(f"Ingestion complete. {nobj_total:,} objects written.", flush=True)
