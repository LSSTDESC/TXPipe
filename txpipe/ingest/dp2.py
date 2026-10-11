"""
TXPipe ingestion stage for the real DP2 catalog produced by dp2_preproc.py
(see dp2_clustering/pipeline/dp2_preproc.py in
github.com/LSSTDESC/dp2_clustering), as opposed to the
Cardinal/Flagship mock catalogs handled by dp2_mocks.py.

dp2_preproc.py writes a single consolidated FITS table (cat_basic.fits),
already trimmed to the requested SNR/maglim/star-galaxy selection, with
dereddened cModel and GAaP photometry. Unlike the mocks, there is no
truth or photo-z redshift column at all.  To let this catalog run through
the existing lens-selection stages unmodified (TXTruthLensSelector /
TXTruthLensCatalogSplitter), this stage writes a constant dummy value as
`redshift_true`; the accompanying pipeline config uses a single wide
`lens_zbin_edges` bin spanning that value, i.e. no real tomographic
binning.

Expected input FITS columns (per dp2_preproc.py's get_catalog_columns /
process_tract):
    coord_ra, coord_dec, ebv
    {band}_cModelFlux, {band}_cModelFluxErr, {band}_cModelFlux_dered,
    {band}_cModelMag_dered   (for each band in `bands`)
    r_extendedness, i_extendedness

coord_ra/coord_dec are already in degrees (dp2_preproc.py applies no
radian conversion and compares them directly against a degree-valued
RA/Dec box).

Output HDF5 layout (mirrors the TXPipe photometry_catalog convention used
by txpipe/ingest/dp2_mocks.py):
/photometry/
    ra, dec, pixel                         (pixel: RING order, given nside)
    mag_{band}, mag_{band}_err, mag_err_{band}   (per band; both error
                                                   naming conventions are
                                                   written, matching
                                                   dp2_mocks.py's ERR_ALIASES)
    redshift_true                          (constant dummy value)
    ebv, r_extendedness, i_extendedness
    mag_g                                  (NaN placeholder -- see below)

`mag_g` is written as NaN: dp2_preproc.py's catalog has no g-band
photometry (see `catalog.bands_keep` in params.yml), but
TXBaseLensSelector subclasses unconditionally request a "mag_g" column
regardless of `selection_type`, even though the maglim selection this
pipeline uses never actually reads it.
"""

from ..base_stage import PipelineStage
from ..data_types import HDFFile
from ceci.config import StageParameter
import numpy as np


class TXIngestDP2Catalog(PipelineStage):
    """Ingest the real-DP2 cat_basic.fits catalog (from dp2_preproc.py) into
    a TXPipe-compatible HDF5 photometry catalog.

    Unlike the dp2_mocks ingestion stages, the source file is a single FITS
    table (not chunked parquet) that has already been trimmed down by
    dp2_preproc.py's own selection cuts, so it is read in one go with
    astropy; only the HDF5 write is chunked, for memory headroom.

    Parameters
    ----------
    fits_path : str
        Path to cat_basic.fits produced by dp2_preproc.py.
    bands : list[str]
        Bands to carry over (must match dp2_preproc.py's catalog.bands_keep).
    nside : int
        HEALPix nside for the pixel column (RING ordering, matching every
        TXPipe stage that reads a mask/map -- see MapsFile.read_healpix()).
    dummy_redshift : float
        Constant value written as redshift_true, since no truth or photo-z
        redshift is available for real DP2 data.
    chunk_rows : int
        Rows to write per HDF5 chunk (default 100_000).
    """

    name = "TXIngestDP2Catalog"
    parallel = False

    inputs = []

    outputs = [
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "fits_path": StageParameter(
            str, msg="Path to cat_basic.fits produced by dp2_preproc.py."
        ),
        "bands": StageParameter(
            list, ["r", "i", "z"],
            msg="Bands to carry over (must match dp2_preproc.py's catalog.bands_keep).",
        ),
        "nside": StageParameter(
            int, 1024, msg="HEALPix nside for the pixel column (RING ordering)."
        ),
        "dummy_redshift": StageParameter(
            float, 0.5,
            msg="Constant value written as redshift_true (no truth/photo-z is available).",
        ),
        "chunk_rows": StageParameter(int, 100_000, msg="Rows to write per HDF5 chunk."),
    }

    def run(self):
        from astropy.table import Table
        import healpy as hp

        fits_path = self.config["fits_path"]
        bands = self.config["bands"]
        nside = self.config["nside"]
        chunk_rows = self.config["chunk_rows"]

        print(f"TXIngestDP2Catalog: reading {fits_path}", flush=True)
        cat = Table.read(fits_path, format="fits")
        nobj = len(cat)
        print(f"  {nobj:,} objects", flush=True)

        ra = np.asarray(cat["coord_ra"], dtype=np.float64)
        dec = np.asarray(cat["coord_dec"], dtype=np.float64)
        # RING order, matching hp.ang2pix(nside, ra, dec, lonlat=True) as used
        # (implicitly, with no nest= argument) by every TXPipe mask/map reader.
        pixel = hp.ang2pix(nside, ra, dec, lonlat=True, nest=False).astype(np.int64)

        columns = {
            "ra": ra,
            "dec": dec,
            "pixel": pixel,
            "redshift_true": np.full(nobj, self.config["dummy_redshift"], dtype=np.float64),
            "ebv": np.asarray(cat["ebv"], dtype=np.float64),
            "r_extendedness": np.asarray(cat["r_extendedness"], dtype=np.float64),
            "i_extendedness": np.asarray(cat["i_extendedness"], dtype=np.float64),
            # dp2_preproc.py's catalog.bands_keep never includes g-band (see
            # params.yml), so there is no real g-band photometry to write.
            # TXBaseLensSelector subclasses unconditionally request "mag_g"
            # in their column list regardless of selection_type (see
            # lens_selector.py's data_iterator methods), even though
            # select_lens_maglim -- the selection this pipeline uses -- never
            # actually reads it. Rather than re-running the (expensive,
            # Butler-based) catalog build just to fetch unused g-band
            # photometry, write it as NaN so the column exists without
            # silently implying real data.
            "mag_g": np.full(nobj, np.nan, dtype=np.float64),
        }

        for b in bands:
            flux = np.asarray(cat[f"{b}_cModelFlux"], dtype=np.float64)
            flux_err = np.asarray(cat[f"{b}_cModelFluxErr"], dtype=np.float64)
            # Fractional flux error is unchanged by dereddening (a fixed
            # multiplicative correction cancels in the ratio), so this gives
            # the correct error for the dereddened magnitude below without
            # needing a separate {b}_cModelFluxErr_dered column.
            safe_flux = np.where(flux > 0, flux, np.nan)
            mag_err = (2.5 / np.log(10)) * np.abs(flux_err / safe_flux)

            columns[f"mag_{b}"] = np.asarray(cat[f"{b}_cModelMag_dered"], dtype=np.float64)
            columns[f"mag_{b}_err"] = mag_err
            columns[f"mag_err_{b}"] = mag_err

        output_file = self.open_output("photometry_catalog")
        phot = output_file.create_group("photometry")
        phot.attrs["nobj"] = nobj
        phot.attrs["bands"] = bands

        for col_name, data in columns.items():
            dtype = np.int64 if col_name == "pixel" else np.float64
            dset = phot.create_dataset(col_name, shape=(nobj,), dtype=dtype)
            for start in range(0, nobj, chunk_rows):
                end = min(start + chunk_rows, nobj)
                dset[start:end] = data[start:end]

        output_file.flush()
        output_file.close()
        print(f"TXIngestDP2Catalog: wrote {nobj:,} objects.", flush=True)
