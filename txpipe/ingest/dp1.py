from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection
from .lsst import process_photometry_data, process_shear_data
import numpy as np



# From https://rtn-095.lsst.io/
# The three main fields we want for cosmology are
# the ones in the wide-fast-deep region, excluding
# the globular cluster field and nebula field. That leaves:
#   Euclid Deep Field South
#   Extended Chandra Deep Field South
#   Low Galactic Latitude Field aka Rubin_SV_095_-25

# The tract values are listed in table 2 of that paper:
DP1_COSMOLOGY_FIELDS = [
    "EDFS",
    "ECDFS",
    "LGLF",
]


DP1_TRACTS = {
    # Euclid Deep Field South
    "EDFS": [2393, 2234, 2235, 2394],

    # Extended Chandra Deep Field South
    "ECDFS": [5062, 5063, 5064, 4848, 4849], 

    # Low Galactic Latitude Field / Rubin_SV_095_-25
    "LGLF": [5305, 5306, 5525, 5526],

    # Fornax Dwarf Spheroidal Galaxy
    "FDSG": [4016, 4217, 4218, 4017],

    # Low Ecliptic Latitude Field / Rubin_SV_38_7
    "LELF": [10464, 10221, 10222, 10704, 10705, 10463],

    # Seagull Nebula
    "Seagull": [7850, 7849, 7610, 7611],

    # 47 Tuc Globular Cluster
    "47Tuc": [531, 532, 453, 454],
}

DP1_COSMOLOGY_TRACTS = sum([DP1_TRACTS[_field] for _field in DP1_COSMOLOGY_FIELDS], [])
ALL_TRACTS = sum(DP1_TRACTS.values(), [])


# In case useful later:
DP1_FIELD_CENTERS = {
    "47 Tuc Globular Cluster": (6.02, -72.08),
    "Low Ecliptic Latitude Field": (37.86, 6.98),
    "Fornax Dwarf Spheroidal Galaxy": (40.00, -34.45),
    "Extended Chandra Deep Field South": (53.13, -28.10),
    "Euclid Deep Field South": (59.10, -48.73),
    "Low Galactic Latitude Field": (95.00, -25.00),
    "Seagull Nebula": (106.23, -10.51),
}


DP1_SURVEY_PROPERTIES = {
    "deepCoadd_exposure_time_consolidated_map_sum": "Total exposure time accumulated per sky position (second)",
    "deepCoadd_epoch_consolidated_map_min": "Earliest observation epoch (MJD)",
    "deepCoadd_epoch_consolidated_map_max":  "Latest observation epoch (MJD)",
    "deepCoadd_epoch_consolidated_map_mean": "Mean observation epoch (MJD)",
    "deepCoadd_psf_size_consolidated_map_weighted_mean": "Weighted mean of PSF characteristic width as computed from the determinant radius (pixel)",
    "deepCoadd_psf_e1_consolidated_map_weighted_mean": "Weighted mean of PSF ellipticity component e1",
    "deepCoadd_psf_e2_consolidated_map_weighted_mean": "Weighted mean of PSF ellipticity component e2",
    "deepCoadd_psf_maglim_consolidated_map_weighted_mean": "Weighted mean of PSF flux 5σ magnitude limit (magAB)",
    "deepCoadd_sky_background_consolidated_map_weighted_mean": "Weighted mean of background light level from the sky (nJy)",
    "deepCoadd_sky_noise_consolidated_map_weighted_mean": "Weighted mean of standard deviation of the sky level (nJy)",
    "deepCoadd_dcr_dra_consolidated_map_weighted_mean": "Weighted mean of DCR-induced astrometric shift in right ascension direction, expressed as a proportionality factor",
    "deepCoadd_dcr_ddec_consolidated_map_weighted_mean": "Weighted mean of DCR-induced astrometric shift in declination direction, expressed as a proportionality factor",
    "deepCoadd_dcr_e1_consolidated_map_weighted_mean": "Weighted mean of DCR-induced change in PSF ellipticity (e1), expressed as a proportionality factor",
    "deepCoadd_dcr_e2_consolidated_map_weighted_mean": "Weighted mean of DCR-induced change in PSF ellipticity (e2), expressed as a proportionality factor",
}


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
        ("survey_property_maps", FileCollection),
    ]
    config_options = {
        "butler_config_file": "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
        "cosmology_tracts_only": True,
        "select_field": "",  # If set, only select objects in this field. Overrides cosmology_tracts_only.
    }

    def run(self):
        from lsst.daf.butler import Butler

        # Configure and create the butler. There seem to be several ways
        # to do this, and there is a central collective butler yaml file
        # on NERSC
        butler_config_file = self.config["butler_config_file"]
        butler = Butler(butler_config_file, collections="LSSTComCam/DP1")

        if self.config["select_field"]:
            selected_tracts = DP1_TRACTS[self.config["select_field"]]
        elif self.config["cosmology_tracts_only"]:
            selected_tracts = DP1_COSMOLOGY_TRACTS
        else:
            selected_tracts = ALL_TRACTS


        self.ingest_photometry(butler, selected_tracts)
        self.ingest_visits(butler, selected_tracts)
        self.ingest_survey_property_maps(butler, selected_tracts)


    def ingest_photometry(self, butler, tracts):
        from ..utils.hdf_tools import h5py_shorten, repack
        columns = [
            'objectId',
            'tract',
            'patch',
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
            'z_cModel_flag',
            'deblend_skipped',
            'deblend_failed',

        ]
        n = self.get_catalog_size(butler, "object")

        created_files = False
        photo_start = 0
        shear_start = 0
        data_set_refs = butler.query_datasets("object")
        n_chunks = len(data_set_refs)


        for i, ref in enumerate(data_set_refs):
            tract = ref.dataId["tract"]
            if tract not in tracts:
                print(f"Skipping chunk {i+1} / {n_chunks} since tract {tract} is not selected")
                continue

            d = butler.get("object", dataId=ref.dataId, parameters={'columns': columns})
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"Skipping chunk {i+1} / {n_chunks} since it is empty")
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
                shear_outfile["shear"].attrs["catalog_type"] = "simple"

            # Output these chunks to the output files
            photo_end = photo_start + len(photo_data['ra'])
            shear_end = shear_start + len(shear_data['ra'])
            self.write_output(photo_outfile, "photometry", photo_data, photo_start, photo_end)
            self.write_output(shear_outfile, "shear", shear_data, shear_start, shear_end)

            print(f"Processing chunk {i+1} / {n_chunks} into rows {photo_start:,} - {photo_end:,}")
            photo_start = photo_end
            shear_start = shear_end

        print(f"Final selected objects: {photo_end:,} in photometry and {shear_end:,} in shear")

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

    def ingest_survey_property_maps(self, butler, selected_tracts):
        import healpy
        skymap = butler.get("skyMap")

        map_types = butler.registry.queryDatasetTypes(expression="*consolidated_map*")

        f: FileCollection = self.open_output("survey_property_maps", wrapper=True)
        filenames = []

        for map_type in map_types:
            for band in "ugrizy":
                # Read this map from the butler
                m = butler.get(map_type, band=band)

                # get the tract for each pixel
                ra, dec = healpy.pix2ang(m.nside_sparse, m.valid_pixels, nest=True, lonlat=True)
                tract = skymap.findTractIdArray(ra, dec, degrees=True)

                # filter out pixels not in our selected traccts
                select = np.isin(tract, selected_tracts)
                pixels_to_remove = m.valid_pixels[~select]
                m.update_values_pix(pixels_to_remove, None)

                filename = f.path_for_file(map_type + "_" + band + ".fits")
                m.write(filename, clobber=True)
                filenames.append(filename)

        f.write_listing(filenames)
        f.close()



    def ingest_visits(self, butler, selected_tracts):

        skymap = butler.get("skyMap")

        # There aren't that many columns, we can just dump the whole thing
        with self.open_output("exposures") as f:
            d1 = butler.get("visit_table")

            # Filter the visits to only those in the selected tracts
            ra = d1["ra"]
            dec = d1["dec"]
            tract = skymap.findTractIdArray(ra, dec, degrees=True)
            d1 = d1[np.isin(tract, selected_tracts)]

            g = f.create_group("visits")
            for col in d1.columns:
                data = sanitize(d1[col])
                g.create_dataset(col, data=data)

            # Let's also save the detector visits table as we can use
            # if for null tests on chip center tangential shear.
            g = f.create_group("detector_visits")
            d2 = butler.get("visit_detector_table")

            # Also cut down to the selected tracts
            ra = d2["ra"]
            dec = d2["dec"]
            tract = skymap.findTractIdArray(ra, dec, degrees=True)
            d2 = d2[np.isin(tract, selected_tracts)]
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
            # replace masked values with nans
            if np.ma.isMaskedArray(col):
                col = col.filled(np.nan)
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
