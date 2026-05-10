"""
TXPipe pipeline stage to ingest one or more parquet simulation catalogs
(e.g. from a CosmoDC2-like simulation) and write them out as a single
TXPipe-compatible HDF5 photometry catalog.

Multiple files are concatenated in the order they are listed.  Each file
is streamed in chunks so peak memory is bounded by ``chunk_rows``, not by
the total catalog size.

Expected input parquet columns (per file)
-----------------------------------------
Photometry (LSST bands):
    mag_u_lsst, mag_g_lsst, mag_r_lsst,
    mag_i_lsst, mag_z_lsst, mag_y_lsst
    mag_u_lsst_err, mag_g_lsst_err, mag_r_lsst_err,
    mag_i_lsst_err, mag_z_lsst_err, mag_y_lsst_err

Photometry (Euclid NISP + VIS bands):
    mag_y_euclid_nisp, mag_j_euclid_nisp, mag_h_euclid_nisp,
    mag_vis_euclid

Astrometry / identifiers:
    ra, dec, halo_id, galaxy_id, pixel

Physical properties:
    log_sfr, log_stellar_mass, lm_halo, redshift

Morphology:
    totalHalfLightRadiusArcsec, major, minor

Output HDF5 layout (mirrors TXPipe photometry_catalog convention)
-----------------------------------------------------------------
/photometry/
    ra, dec
    mag_u, mag_g, mag_r, mag_i, mag_z, mag_y          (LSST)
    mag_u_err, mag_g_err, mag_r_err,
    mag_i_err, mag_z_err, mag_y_err
    mag_y_euclid_nisp, mag_j_euclid_nisp,
    mag_h_euclid_nisp, mag_vis_euclid
    redshift_true
    halo_id, galaxy_id, pixel
    log_sfr, log_stellar_mass, lm_halo
    totalHalfLightRadiusArcsec, major, minor

Configuration
-------------
Provide either:

  (a) a single file path:
        parquet_paths: /path/to/catalog.parquet

  (b) a list of file paths (YAML list syntax):
        parquet_paths:
          - /path/to/catalog_part1.parquet
          - /path/to/catalog_part2.parquet
          - /path/to/catalog_part3.parquet

  (c) a glob pattern (resolved alphabetically):
        parquet_paths: /path/to/catalog_*.parquet

All files must share exactly the same column schema.
"""

from ..base_stage import PipelineStage
from ..data_types import HDFFile
from ceci.config import StageParameter
import numpy as np


# ---------------------------------------------------------------------------
# Mapping from parquet column names  →  HDF5 dataset names inside /photometry
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

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
        """Raise ``KeyError`` listing every missing required column in *path*."""
        import pyarrow.parquet as pq
        schema_names = set(pq.read_schema(path).names)
        missing = [c for c in PARQUET_COLUMNS if c not in schema_names]
        if missing:
            raise KeyError(
                f"File {path!r} is missing required columns: {missing}\n"
                f"Available columns in that file: {sorted(schema_names)}"
            )

    def _iter_file_chunks(self, path):
        """Yield ``(batch_size, pandas.DataFrame)`` for every chunk of *path*."""
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=self.config["chunk_rows"],
            columns=PARQUET_COLUMNS,
        ):
            yield len(batch), batch.to_pandas()

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