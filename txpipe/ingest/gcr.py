from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, HDFFile
from ..utils import band_variants, metacal_variants, moments_to_shear
from ceci.config import StageParameter
import numpy as np
import glob
import re


class TXMetacalGCRInput(PipelineStage):
    """
    Ingest metacal catalogs from GCRCatalogs

    This loads a matched shear and photometry catalog.

    """

    name = "TXMetacalGCRInput"
    parallel = False
    inputs = []

    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "cat_name": StageParameter(str, "", msg="Name of the GCR catalog to load."),
        "single_tract": StageParameter(str, "", msg="Single tract to use (optional)."),
        "length": StageParameter(int, 0, msg="Pre-known length, if the catalog has been checked at previously."),
        "table_dir": StageParameter(str, "", msg="Directory for table files (optional)."),
        "data_release": StageParameter(str, "", msg="Data release identifier (optional)."),
    }

    def run(self):
        import GCRCatalogs
        import GCR
        import h5py

        # Open input data.  We do not treat this as a formal "input"
        # since it's the starting point of the whol pipeline and so is
        # not in a TXPipe format.
        cat_name = self.config["cat_name"]

        # If set, override some keys from the config
        config_overwrite = {}
        for key in ["table_dir", "data_release"]:
            if self.config[key]:
                config_overwrite[key] = self.config[key]

        print(f"Loading catalog {cat_name} + {config_overwrite}")
        cat = GCRCatalogs.load_catalog(cat_name, config_overwrite=config_overwrite)
        # Total size is needed to set up the output file,
        # although in larger files it is a little slow to compute this.
        if self.config["length"] == 0:
            n = len(cat)
            print(f"Total catalog size = {n}")
        else:
            n = self.config["length"]
            print(f"Using fixed specified size = {n}")

        # This option, which prevented memory leaks with a previous
        # catalog format, appears not to work for Parquet catalogs,
        # but the memory leaks don't happen in that case either.
        try:
            cat.master.use_cache = False
        except AttributeError:
            pass

        available = cat.list_all_quantities()
        bands = []
        for b in "ugrizy":
            if f"mcal_mag_{b}" in available:
                bands.append(b)

        # Columns that we will need.
        shear_cols = (
            [
                "id",
                "ra",
                "dec",
                "mcal_psf_g1",
                "mcal_psf_g2",
                "mcal_psf_T_mean",
                "mcal_flags",
            ]
            + metacal_variants("mcal_g1", "mcal_g2", "mcal_T", "mcal_s2n")
            + band_variants(
                bands, "mcal_mag", "mcal_mag_err", shear_catalog_type="metacal"
            )
        )

        # Input columns for photometry
        photo_cols = ["id", "ra", "dec", "extendedness", "tract"]

        # Photometry columns (non-metacal)
        for band in "ugrizy":
            photo_cols.append(f"mag_{band}")
            photo_cols.append(f"magerr_{band}")
            photo_cols.append(f"snr_{band}_cModel")

        # For shear we just add a weight column, and the non-rounded PSF estimates
        shear_out_cols = shear_cols + ["weight", "psf_g1", "psf_g2"]

        # We want these in the input but not the output as we construct
        # other values from them instead
        shear_cols += ["IxxPSF", "IxyPSF", "IyyPSF"]

        # For the photometry output we strip off the _cModeel suffix.
        photo_out_cols = [
            col[:-7] if col.endswith("_cModel") else col for col in photo_cols
        ]

        # eliminate duplicates before loading
        cols = list(set(shear_cols + photo_cols))

        start = 0
        shear_output = None
        photo_output = None

        # Loop through the data, as chunke natively by GCRCatalogs
        single_tract = self.config["single_tract"]

        if single_tract:
            kwargs = {"native_filters": f"tract == {single_tract}"}
            print(f"Selecting one tract only: {single_tract}")
        else:
            kwargs = {}

        for data in cat.get_quantities(cols, return_iterator=True, **kwargs):
            # Some columns have different names in input than output
            self.rename_columns(data)
            self.add_weight_column(data)

            # First chunk of data we use to set up the output
            # It is easier this way (no need to check types etc)
            # if we change the column list
            if shear_output is None:
                shear_output = self.setup_output(
                    "shear_catalog", "shear", data, shear_out_cols, n
                )
                photo_output = self.setup_output(
                    "photometry_catalog", "photometry", data, photo_out_cols, n
                )

            # Write out this chunk of data to HDF
            end = start + len(data["ra"])
            print(f"    Saving {start} - {end}")
            self.write_output(shear_output, "shear", shear_out_cols, start, end, data)
            self.write_output(
                photo_output, "photometry", photo_out_cols, start, end, data
            )
            start = end

        # All done!
        photo_output.close()
        shear_output.close()

    def rename_columns(self, data):
        for band in "ugrizy":
            data[f"snr_{band}"] = data[f"snr_{band}_cModel"]
            del data[f"snr_{band}_cModel"]

        Ixx = data["IxxPSF"]
        Ixy = data["IxyPSF"]
        Iyy = data["IyyPSF"]
        data["psf_g1"], data["psf_g2"] = moments_to_shear(Ixx, Iyy, Ixy)

    def setup_output(self, name, group, cat, cols, n):
        import h5py

        f = self.open_output(name)
        g = f.create_group(group)
        g.attrs["bands"] = "ugrizy"

        for name in cols:
            g.create_dataset(name, shape=(n,), dtype=cat[name].dtype)
        return f

    def add_weight_column(self, data):
        n = len(data["ra"])
        data["weight"] = np.ones(n)

    def write_output(self, output_file, group_name, cols, start, end, data):
        g = output_file[group_name]
        for name in cols:
            g[name][start:end] = data[name]


class TXIngestStars(PipelineStage):
    """
    Ingest a star catalog from GCRCatalogs

    Includes shape information (i.e. PSF samples) and whether the star was used
    in PSF estimation.
    """
    name = "TXIngestStars"
    parallel = False
    inputs = []

    outputs = [
        ("star_catalog", HDFFile),
    ]
    config_options = {
        "single_tract": StageParameter(str, "", msg="Single tract to use (optional)."),
        "cat_name": StageParameter(str, "", msg="Name of the GCR catalog to load."),
        "length": StageParameter(int, 0, msg="Pre-known length, if the catalog has been checked at previously."),
    }

    def run(self):
        import GCRCatalogs
        import GCR
        import h5py
        from ..utils.hdf_tools import repack, h5py_shorten

        cat_name = self.config["cat_name"]
        cat = GCRCatalogs.load_catalog(cat_name)

        # This is the max possible length of the stars.
        # Actually much smaller of course
        if self.config["length"]:
            n = self.config["length"]
            print(f"Using fixed size {n}")
        else:
            n = len(cat)

        print(f"Full catalog size = {n}")
        # Columns we need to load in for the star data -
        # the measured object moments and the identifier telling us
        # if it was used in PSF measurement
        star_cols = [
            "id",
            "ra",
            "dec",
            "calib_psf_used",
            "calib_psf_reserved",
            "extendedness",
            "tract",
            "mag_u",
            "mag_g",
            "mag_r",
            "mag_i",
            "mag_z",
            "mag_y",
            "Ixx",
            "Ixy",
            "Iyy",
            "IxxPSF",
            "IxyPSF",
            "IyyPSF",
        ]

        # The star output names are mostly different to the input names
        star_out_cols = [
            # These are read directly
            "id",
            "ra",
            "dec",
            "calib_psf_used",
            "calib_psf_reserved",
            "extendedness",
            "tract",
            "mag_u",
            "mag_g",
            "mag_r",
            "mag_i",
            "mag_z",
            "mag_y",
            # These are calculated
            "measured_e1",
            "measured_e2",
            "model_e1",
            "model_e2",
            "measured_T",
            "model_T",
        ]

        single_tract = self.config["single_tract"]

        if single_tract:
            kwargs = {"native_filters": f"tract == {single_tract}"}
            print(f"Selecting one tract only: {single_tract}")
        else:
            kwargs = {}

        # As with the galaxy ingestion, this option doesn't
        # work with Parquet catalogs.
        try:
            cat.master.use_cache = False
        except AttributeError:
            pass

        start = 0
        star_start = 0
        star_output = None
        for data in cat.get_quantities(star_cols, return_iterator=True, **kwargs):
            end = start + len(data["ra"])
            print(f"Reading data {start:,} - {end:,}")
            # Some columns have different names in input than output
            star_data = self.compute_star_data(data)
            star_end = star_start + len(star_data["ra"])
            if star_output is None:
                star_output = self.setup_output(
                    "star_catalog", "stars", star_data, star_out_cols, n
                )
            self.write_output(
                star_output, "stars", star_out_cols, star_start, star_end, star_data
            )

            start = end
            star_start = star_end

        # Cut down to just include stars.
        for col in star_out_cols:
            h5py_shorten(star_output["stars"], col, star_end)

        star_output.close()

        # Run h5repack on the file
        repack(self.get_output("star_catalog"))

    def setup_output(self, name, group, cat, cols, n):
        import h5py

        f = self.open_output(name)
        g = f.create_group(group)
        for name in cols:
            g.create_dataset(name, shape=(n,), dtype=cat[name].dtype)
        return f

    def write_output(self, output_file, group_name, cols, start, end, data):
        g = output_file[group_name]
        for name in cols:
            g[name][start:end] = data[name]

    def compute_star_data(self, data):
        star_data = {}
        # We specifically use the stars chosen for PSF measurement
        star = data["calib_psf_used"] | data["calib_psf_reserved"]

        for col in [
            "id",
            "ra",
            "dec",
            "calib_psf_used",
            "calib_psf_reserved",
            "extendedness",
            "tract",
        ]:
            star_data[col] = data[col][star]

        for b in "ugrizy":
            star_data[f"mag_{b}"] = data[f"mag_{b}"][star]

        # HSM reports moments.  We convert these into
        # ellipticities.  We do this for both the star shape
        # itself and the PSF model.
        kinds = [("", "measured_"), ("PSF", "model_")]

        for in_name, out_name in kinds:
            # Pulling out the correct moment columns
            Ixx = data[f"Ixx{in_name}"][star]
            Iyy = data[f"Iyy{in_name}"][star]
            Ixy = data[f"Ixy{in_name}"][star]

            # Conversion of moments to e1, e2
            T = Ixx + Iyy
            e1, e2 = moments_to_shear(Ixx, Iyy, Ixy)

            # save to output
            star_data[f"{out_name}e1"] = e1
            star_data[f"{out_name}e2"] = e2
            star_data[f"{out_name}T"] = T

        return star_data





        




# response to an old Stack Overflow question of mine:
# https://stackoverflow.com/questions/33529057/indices-that-intersect-and-sort-two-numpy-arrays
def intersecting_indices(x, y):
    u_x, u_idx_x = np.unique(x, return_index=True)
    u_y, u_idx_y = np.unique(y, return_index=True)
    i_xy = np.intersect1d(u_x, u_y, assume_unique=True)
    i_idx_x = u_idx_x[np.in1d(u_x, i_xy, assume_unique=True)]
    i_idx_y = u_idx_y[np.in1d(u_y, i_xy, assume_unique=True)]
    return i_idx_x, i_idx_y

