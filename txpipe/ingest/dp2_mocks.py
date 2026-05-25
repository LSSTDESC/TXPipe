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
from ..data_types import HDFFile, MapsFile
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


# ---------------------------------------------------------------------------
# Truth catalog ingestion
# ---------------------------------------------------------------------------
#
# The "gold" truth parquet files (e.g. dp2_mock_run_flagship_gold/) differ
# from the observed-condition files in two important ways:
#
#   1. No magnitude-error columns – photometry is noise-free simulation truth.
#      Magnitude errors are therefore synthesised as zeros.
#
#   2. No pre-computed ``pixel`` column – HEALPix pixel indices are computed
#      on-the-fly from (ra, dec) using healpy.
#
# Additional truth-only columns (true shear, ellipticities, detailed
# morphology) are also written and can be used for validation.
# ---------------------------------------------------------------------------

# Parquet columns that exist in the truth file and map directly to HDF5
TRUTH_COLUMN_MAP = {
    # LSST magnitudes (noise-free truth)
    "mag_u_lsst": "mag_u",
    "mag_g_lsst": "mag_g",
    "mag_r_lsst": "mag_r",
    "mag_i_lsst": "mag_i",
    "mag_z_lsst": "mag_z",
    "mag_y_lsst": "mag_y",
    # Euclid magnitudes
    "mag_y_euclid_nisp": "mag_y_euclid_nisp",
    "mag_j_euclid_nisp": "mag_j_euclid_nisp",
    "mag_h_euclid_nisp": "mag_h_euclid_nisp",
    "mag_vis_euclid": "mag_vis_euclid",
    # Astrometry
    "ra": "ra",
    "dec": "dec",
    # Identifiers
    "halo_id": "halo_id",
    "galaxy_id": "galaxy_id",
    # Physical properties
    "log_sfr": "log_sfr",
    "log_stellar_mass": "log_stellar_mass",
    "lm_halo": "lm_halo",
    "redshift": "redshift_true",
    # Morphology
    "totalHalfLightRadiusArcsec": "totalHalfLightRadiusArcsec",
    "major": "major",
    "minor": "minor",
    # Truth-only: weak lensing shear and intrinsic ellipticity
    "gamma1": "gamma1",
    "gamma2": "gamma2",
    "eps1_gal": "eps1_gal",
    "eps2_gal": "eps2_gal",
    # Truth-only: detailed bulge/disk morphology
    "bulge_r50": "bulge_r50",
    "disk_r50": "disk_r50",
    "bulge_fraction": "bulge_fraction",
    "orientationAngle": "orientationAngle",
}

# Columns with integer HDF5 dtype (everything else is float64)
_TRUTH_INT_COLS = {"halo_id", "galaxy_id"}

# LSST bands for which we synthesise zero magnitude errors
_TRUTH_BANDS = ("u", "g", "r", "i", "z", "y")

# Parquet columns to read from each file
TRUTH_PARQUET_COLUMNS = list(TRUTH_COLUMN_MAP.keys())


class TXIngestTruthParquetCatalog(PipelineStage):
    """Ingest one or more *truth* (noise-free) Flagship/DP2 parquet files into
    a single TXPipe-compatible HDF5 photometry catalog.

    Key differences from ``TXIngestParquetCatalog`` (the observed-condition
    ingestion stage):

    * **No photometric errors in the input files.**  Magnitude errors are
      synthesised as zeros in the output, satisfying the ``mag_u_err`` /
      ``mag_err_u`` layout expected by downstream TXPipe stages.

    * **No pre-computed pixel column.**  HEALPix pixel indices are computed
      from (ra, dec) using healpy at the resolution set by ``nside``.

    * **Extra truth-only columns** (true shear γ₁/γ₂, intrinsic ellipticities
      ε₁/ε₂, bulge/disk morphology) are written to the output for later
      validation use.

    Parameters
    ----------
    parquet_paths : str or list[str]
        A single path, a YAML list of paths, or a glob pattern.
    chunk_rows : int
        Rows to read per chunk (default 100 000).
    nside : int
        HEALPix nside used to compute the ``pixel`` column (default 512).
    """

    name = "TXIngestTruthParquetCatalog"
    parallel = False

    inputs = []

    outputs = [
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "parquet_paths": StageParameter(
            str,
            msg=(
                "Path(s) to input truth parquet files. "
                "Accepts a single path, a YAML list of paths, or a glob pattern."
            ),
        ),
        "chunk_rows": StageParameter(
            int, 100_000, msg="Rows to read per chunk per file"
        ),
        "nside": StageParameter(
            int, 512, msg="HEALPix nside for pixel-index computation"
        ),
    }

    # ------------------------------------------------------------------
    # Private helpers  (mirrors TXIngestParquetCatalog where possible)
    # ------------------------------------------------------------------

    def _resolve_paths(self):
        import glob as _glob
        import ast

        raw = self.config["parquet_paths"]

        if isinstance(raw, list):
            paths = [str(p) for p in raw]
        elif isinstance(raw, str) and raw.strip().startswith('['):
            paths = [str(p) for p in ast.literal_eval(raw)]
        else:
            expanded = sorted(_glob.glob(raw))
            paths = expanded if expanded else [str(raw)]

        if not paths:
            raise ValueError(
                f"No parquet files found for pattern/path: {raw!r}"
            )
        return paths

    def _count_rows(self, path):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows

    def _validate_columns(self, path):
        import pyarrow.parquet as pq
        schema_names = set(pq.read_schema(path).names)
        missing = [c for c in TRUTH_PARQUET_COLUMNS if c not in schema_names]
        if missing:
            raise KeyError(
                f"File {path!r} is missing required columns: {missing}\n"
                f"Available columns: {sorted(schema_names)}"
            )

    def _iter_file_chunks(self, path):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=self.config["chunk_rows"],
            columns=TRUTH_PARQUET_COLUMNS,
        ):
            yield len(batch), batch.to_pandas()

    def _compute_pixels(self, ra, dec):
        import healpy as hp
        nside = self.config["nside"]
        return hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        import pyarrow.parquet as pq  # verify available

        paths = self._resolve_paths()
        n_files = len(paths)
        print(
            f"TXIngestTruthParquetCatalog: ingesting {n_files} truth file(s).",
            flush=True,
        )

        # 1. Validate columns and count rows
        file_row_counts = []
        for i, path in enumerate(paths):
            print(f"  [{i+1}/{n_files}] Checking: {path}", flush=True)
            self._validate_columns(path)
            n = self._count_rows(path)
            file_row_counts.append(n)
            print(f"           {n:,} rows", flush=True)

        nobj_total = sum(file_row_counts)
        print(f"  Grand total: {nobj_total:,} rows.", flush=True)

        # 2. Open output and pre-allocate all datasets
        output_file = self.open_output("photometry_catalog")
        phot = output_file.create_group("photometry")
        phot.attrs["nobj"] = nobj_total

        # Primary columns from the parquet column map
        for parq_col, hdf_col in TRUTH_COLUMN_MAP.items():
            dtype = np.int64 if parq_col in _TRUTH_INT_COLS else np.float64
            phot.create_dataset(hdf_col, shape=(nobj_total,), dtype=dtype)

        # pixel column (computed, not in parquet)
        phot.create_dataset("pixel", shape=(nobj_total,), dtype=np.int64)

        # Synthesised zero magnitude errors: mag_u_err convention
        for band in _TRUTH_BANDS:
            phot.create_dataset(f"mag_{band}_err", shape=(nobj_total,), dtype=np.float64)
        # Synthesised zero magnitude errors: mag_err_u convention (alias)
        for band in _TRUTH_BANDS:
            phot.create_dataset(f"mag_err_{band}", shape=(nobj_total,), dtype=np.float64)

        # 3. Stream data into the pre-allocated datasets
        global_row = 0
        for file_idx, (path, file_nrows) in enumerate(
            zip(paths, file_row_counts)
        ):
            file_end = global_row + file_nrows
            print(f"  [{file_idx+1}/{n_files}] Writing: {path}", flush=True)

            chunk_start = global_row
            for n_batch, df in self._iter_file_chunks(path):
                chunk_end = chunk_start + n_batch

                # Write mapped columns
                for parq_col, hdf_col in TRUTH_COLUMN_MAP.items():
                    phot[hdf_col][chunk_start:chunk_end] = df[parq_col].to_numpy()

                # Compute and write pixel indices
                phot["pixel"][chunk_start:chunk_end] = self._compute_pixels(
                    df["ra"].to_numpy(), df["dec"].to_numpy()
                )

                # Write synthesised zero errors (both naming conventions)
                zeros = np.zeros(n_batch, dtype=np.float64)
                for band in _TRUTH_BANDS:
                    phot[f"mag_{band}_err"][chunk_start:chunk_end] = zeros
                    phot[f"mag_err_{band}"][chunk_start:chunk_end] = zeros

                print(
                    f"    rows {chunk_start:,} – {chunk_end-1:,}", flush=True
                )
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
        print(
            f"Truth ingestion complete. {nobj_total:,} objects written.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Flagship DP2 catalog ingestion
# ---------------------------------------------------------------------------
#
# The "small" Flagship DP2 parquet files (e.g. flagship-ramin-*-small.pq)
# differ from the observed-condition dp2 files in several ways:
#
#   1. Position columns are named ra_mag_gal / dec_mag_gal.
#   2. Redshift column is observed_redshift_gal.
#   3. Photometry columns contain fluxes in erg/s/cm²/Hz (AB system),
#      not magnitudes.  Values ~1e-30 correspond to ~26 AB mag.
#      Conversion applied: m = -2.5 * log10(f) - 48.6.
#      Unphysical (<=0) fluxes are replaced with 99.0.
#   4. No pre-computed pixel column – computed from (ra, dec) with healpy.
#   5. No magnitude errors – synthesised as zeros.
# ---------------------------------------------------------------------------

# Mapping: parquet column  →  HDF5 dataset name
FLAGSHIP_COLUMN_MAP = {
    # Astrometry (renamed)
    "ra_mag_gal": "ra",
    "dec_mag_gal": "dec",
    # Redshift (renamed)
    "observed_redshift_gal": "redshift_true",
    # LSST extinction-corrected fluxes → mag_* (converted to AB mag during write)
    "lsst_u_el_model3_ext": "mag_u",
    "lsst_g_el_model3_ext": "mag_g",
    "lsst_r_el_model3_ext": "mag_r",
    "lsst_i_el_model3_ext": "mag_i",
    "lsst_z_el_model3_ext": "mag_z",
    "lsst_y_el_model3_ext": "mag_y",
    # Euclid extinction-corrected fluxes → mag_* (converted to AB mag during write)
    "euclid_nisp_h_el_model3_ext": "mag_h_euclid_nisp",
    "euclid_nisp_j_el_model3_ext": "mag_j_euclid_nisp",
    "euclid_nisp_y_el_model3_ext": "mag_y_euclid_nisp",
    "euclid_vis_el_model3_ext": "mag_vis_euclid",
    # Identifiers
    "halo_id": "halo_id",
    "galaxy_id": "galaxy_id",
    # Shear and intrinsic ellipticity
    "gamma1": "gamma1",
    "gamma2": "gamma2",
    "eps1_gal": "eps1_gal",
    "eps2_gal": "eps2_gal",
    # Morphology
    "disk_r50": "disk_r50",
    "bulge_r50": "bulge_r50",
    "bulge_fraction": "bulge_fraction",
    # MW extinction
    "mw_extinction": "mw_extinction",
}

# Parquet columns whose values are fluxes (erg/s/cm²/Hz) and must be
# converted to AB magnitudes before writing to HDF5.
FLAGSHIP_PHOTOMETRY_COLS = {
    "lsst_u_el_model3_ext", "lsst_g_el_model3_ext", "lsst_r_el_model3_ext",
    "lsst_i_el_model3_ext", "lsst_z_el_model3_ext", "lsst_y_el_model3_ext",
    "euclid_nisp_h_el_model3_ext", "euclid_nisp_j_el_model3_ext",
    "euclid_nisp_y_el_model3_ext", "euclid_vis_el_model3_ext",
}

_FLAGSHIP_INT_COLS = {"halo_id", "galaxy_id"}
_FLAGSHIP_BANDS = ("u", "g", "r", "i", "z", "y")

FLAGSHIP_PARQUET_COLUMNS = list(FLAGSHIP_COLUMN_MAP.keys())


def _flux_to_ab_mag(flux_array):
    """Convert flux in erg/s/cm²/Hz to AB magnitude.  Non-positive fluxes → 99."""
    flux = flux_array.astype(np.float64)
    mag = np.full(len(flux), 99.0, dtype=np.float64)
    good = flux > 0
    mag[good] = -2.5 * np.log10(flux[good]) - 48.6
    return mag


class TXIngestFlagshipCatalog(PipelineStage):
    """Ingest Flagship DP2 parquet files into a TXPipe HDF5 photometry catalog.

    Handles the column-naming differences of the flagship-ramin-* files
    (ra_mag_gal, dec_mag_gal, observed_redshift_gal) and converts the
    extinction-corrected flux columns to AB magnitudes on the fly.

    Parameters
    ----------
    parquet_paths : str or list[str]
        Single path, YAML list, or glob pattern.
    chunk_rows : int
        Rows per read chunk (default 100 000).
    nside : int
        HEALPix nside for pixel-index computation (default 512).
    """

    name = "TXIngestFlagshipCatalog"
    parallel = False
    inputs = []
    outputs = [("photometry_catalog", HDFFile)]

    config_options = {
        "parquet_paths": StageParameter(
            str,
            msg="Path(s) to Flagship parquet files (single, list, or glob).",
        ),
        "chunk_rows": StageParameter(int, 100_000, msg="Rows per chunk"),
        "nside": StageParameter(int, 512, msg="HEALPix nside for pixel indices"),
    }

    # ------------------------------------------------------------------

    def _resolve_paths(self):
        import glob as _glob, ast
        raw = self.config["parquet_paths"]
        if isinstance(raw, list):
            return [str(p) for p in raw]
        if isinstance(raw, str) and raw.strip().startswith('['):
            return [str(p) for p in ast.literal_eval(raw)]
        expanded = sorted(_glob.glob(raw))
        paths = expanded if expanded else [str(raw)]
        if not paths:
            raise ValueError(f"No parquet files found for: {raw!r}")
        return paths

    def _count_rows(self, path):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows

    def _validate_columns(self, path):
        import pyarrow.parquet as pq
        schema_names = set(pq.read_schema(path).names)
        missing = [c for c in FLAGSHIP_PARQUET_COLUMNS if c not in schema_names]
        if missing:
            raise KeyError(
                f"File {path!r} is missing required columns: {missing}\n"
                f"Available: {sorted(schema_names)}"
            )

    def _iter_file_chunks(self, path):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=self.config["chunk_rows"],
            columns=FLAGSHIP_PARQUET_COLUMNS,
        ):
            yield len(batch), batch.to_pandas()

    def _compute_pixels(self, ra, dec):
        import healpy as hp
        return hp.ang2pix(self.config["nside"], ra, dec, lonlat=True, nest=False)

    # ------------------------------------------------------------------

    def run(self):
        import pyarrow.parquet as pq  # noqa: F401 – verify available

        paths = self._resolve_paths()
        n_files = len(paths)
        print(f"TXIngestFlagshipCatalog: ingesting {n_files} file(s).", flush=True)

        file_row_counts = []
        for i, path in enumerate(paths):
            print(f"  [{i+1}/{n_files}] Checking: {path}", flush=True)
            self._validate_columns(path)
            n = self._count_rows(path)
            file_row_counts.append(n)
            print(f"           {n:,} rows", flush=True)

        nobj_total = sum(file_row_counts)
        print(f"  Grand total: {nobj_total:,} rows.", flush=True)

        output_file = self.open_output("photometry_catalog")
        phot = output_file.create_group("photometry")
        phot.attrs["nobj"] = nobj_total

        for parq_col, hdf_col in FLAGSHIP_COLUMN_MAP.items():
            dtype = np.int64 if parq_col in _FLAGSHIP_INT_COLS else np.float64
            phot.create_dataset(hdf_col, shape=(nobj_total,), dtype=dtype)

        phot.create_dataset("pixel", shape=(nobj_total,), dtype=np.int64)

        for band in _FLAGSHIP_BANDS:
            phot.create_dataset(f"mag_{band}_err", shape=(nobj_total,), dtype=np.float64)
            phot.create_dataset(f"mag_err_{band}", shape=(nobj_total,), dtype=np.float64)

        global_row = 0
        for file_idx, (path, file_nrows) in enumerate(zip(paths, file_row_counts)):
            file_end = global_row + file_nrows
            print(f"  [{file_idx+1}/{n_files}] Writing: {path}", flush=True)
            chunk_start = global_row
            for n_batch, df in self._iter_file_chunks(path):
                chunk_end = chunk_start + n_batch

                for parq_col, hdf_col in FLAGSHIP_COLUMN_MAP.items():
                    if parq_col in FLAGSHIP_PHOTOMETRY_COLS:
                        values = _flux_to_ab_mag(df[parq_col].to_numpy())
                    else:
                        values = df[parq_col].to_numpy()
                    phot[hdf_col][chunk_start:chunk_end] = values

                phot["pixel"][chunk_start:chunk_end] = self._compute_pixels(
                    df["ra_mag_gal"].to_numpy(), df["dec_mag_gal"].to_numpy()
                )

                zeros = np.zeros(n_batch, dtype=np.float64)
                for band in _FLAGSHIP_BANDS:
                    phot[f"mag_{band}_err"][chunk_start:chunk_end] = zeros
                    phot[f"mag_err_{band}"][chunk_start:chunk_end] = zeros

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
        print(f"Flagship ingestion complete. {nobj_total:,} objects written.", flush=True)


# ---------------------------------------------------------------------------
# Flagship footprint mask generation
# ---------------------------------------------------------------------------

class TXMaskFromParquet(PipelineStage):
    """Build a HEALPix binary footprint mask from galaxy positions in parquet files.

    Reads only ra_mag_gal and dec_mag_gal, computes which HEALPix pixels are
    hit, and writes a TXPipe-compatible MapsFile mask (all hit pixels = 1).

    Parameters
    ----------
    parquet_paths : str or list[str]
        Same format as TXIngestFlagshipCatalog.
    nside : int
        HEALPix resolution (default 512, ~6.9 arcmin).
    chunk_rows : int
        Rows per read chunk (default 100 000).
    """

    name = "TXMaskFromParquet"
    parallel = False
    inputs = []

    outputs = [("mask", MapsFile)]

    config_options = {
        "parquet_paths": StageParameter(
            str,
            msg="Path(s) to parquet files (single, list, or glob).",
        ),
        "nside": StageParameter(int, 512, msg="HEALPix nside"),
        "chunk_rows": StageParameter(int, 100_000, msg="Rows per chunk"),
    }

    def _resolve_paths(self):
        import glob as _glob, ast
        raw = self.config["parquet_paths"]
        if isinstance(raw, list):
            return [str(p) for p in raw]
        if isinstance(raw, str) and raw.strip().startswith('['):
            return [str(p) for p in ast.literal_eval(raw)]
        expanded = sorted(_glob.glob(raw))
        paths = expanded if expanded else [str(raw)]
        if not paths:
            raise ValueError(f"No parquet files found for: {raw!r}")
        return paths

    def run(self):
        import pyarrow.parquet as pq
        import healpy as hp

        nside = self.config["nside"]
        paths = self._resolve_paths()
        n_files = len(paths)
        print(f"TXMaskFromParquet: building nside={nside} mask from {n_files} file(s).", flush=True)

        hit_pixels = set()
        for i, path in enumerate(paths):
            print(f"  [{i+1}/{n_files}] Reading: {path}", flush=True)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=self.config["chunk_rows"],
                columns=["ra_mag_gal", "dec_mag_gal"],
            ):
                df = batch.to_pandas()
                pix = hp.ang2pix(
                    nside,
                    df["ra_mag_gal"].to_numpy(),
                    df["dec_mag_gal"].to_numpy(),
                    lonlat=True,
                    nest=False,
                )
                hit_pixels.update(pix.tolist())

        pix_arr = np.array(sorted(hit_pixels), dtype=np.int64)
        values = np.ones(len(pix_arr), dtype=np.float64)

        pixel_area_deg2 = hp.nside2pixarea(nside, degrees=True)
        area = len(pix_arr) * pixel_area_deg2
        f_sky = area / 41252.96125
        print(f"  Unmasked pixels: {len(pix_arr):,}", flush=True)
        print(f"  Area: {area:.2f} sq deg  (f_sky = {f_sky:.5f})", flush=True)

        metadata = {
            "pixelization": "healpix",
            "nside": nside,
            "nest": False,
            "area": area,
            "f_sky": f_sky,
        }

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix_arr, values, metadata)

        print("TXMaskFromParquet: mask written.", flush=True)