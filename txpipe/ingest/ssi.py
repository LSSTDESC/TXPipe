from ..base_stage import PipelineStage
from ..data_types import (
    HDFFile,
    FitsFile,
)
from ..utils import (
    nanojansky_to_mag_ab,
    nanojansky_err_to_mag_ab,
)
from .base import TXIngestCatalogBase, TXIngestCatalogFits
import numpy as np
import warnings

# translate between GCR and TXpipe column names
column_names = {
    "coord_ra": "ra",
    "coord_dec": "dec",
}

class TXIngestSSIGCR(TXIngestCatalogBase):
    """
    Ingest SSI catalogs using GCR

    Does not treat the injection or ssi photometry catalogs as formal inputs
    since they are not in a format TXPipe can recognize
    """

    name = "TXIngestSSIGCR"
    parallel = False
    inputs = []

    outputs = [
        ("injection_catalog", HDFFile),
        ("ssi_photometry_catalog", HDFFile),
        ("ssi_uninjected_photometry_catalog", HDFFile),
    ]

    config_options = {
        "injection_catalog_name": "",  # Catalog of objects manually injected
        "ssi_photometry_catalog_name": "",  # Catalog of objects from real data with no injections
        "ssi_uninjected_photometry_catalog_name": "",  # Catalog of objects from real data with no injections
        "GCRcatalog_path": "",
        "flux_name": "gaap3p0Flux",
    }

    def run(self):
        """
        Run the analysis for this stage.

        Loads the catalogs using gcr and saves the relevent columns to a hdf5 format
        that TXPipe can read
        """
        # This is needed to access the SSI runs currently on NERSC run by SRV team
        # As the final runs become more formalized, this could be removed
        if self.config["GCRcatalog_path"] != "":
            import sys

            sys.path.insert(0, self.config["GCRcatalog_path"])
        import GCRCatalogs

        # attempt to ingest three types of catalog
        catalogs_labels = [
            "injection_catalog",
            "ssi_photometry_catalog",
            "ssi_uninjected_photometry_catalog",
        ]

        for catalog_label in catalogs_labels:
            input_catalog_name = self.config[f"{catalog_label}_name"]

            if input_catalog_name == "":
                print(f"No catalog {catalog_label} name provided")
                continue

            output_file = self.open_output(catalog_label)
            group = output_file.create_group("photometry")

            gc0 = GCRCatalogs.load_catalog(input_catalog_name)
            native_quantities = gc0.list_all_native_quantities()

            # iterate over native quantities as some might be missing (or be nans)
            for q in native_quantities:
                try:
                    qobj = gc0.get_quantities(q)
                    # TO DO: load with iterator (currently returns full datatset)
                    # GCRCatalogs.lsst_object.LSSTInjectedObjectTable iterator currently
                    # returns full dataset

                    if np.isnan(qobj[q]).all():
                        continue  # skip the quantities that are empty

                    group.create_dataset(q, data=qobj[q], dtype=qobj[q].dtype)
                    if q in column_names:
                        # also save with TXPipe names
                        group[column_names[q]] = group[q]

                except KeyError:  # skip quantities that are missing
                    warnings.warn(
                        f"quantity {q} was missing from the GCRCatalog object"
                    )
                    continue

                except TypeError:
                    warnings.warn(
                        f"Quantity {q} coud not be saved as it has a data type not recognised by hdf5"
                    )

            # convert fluxes to mags using txpipe/utils/conversion.py
            bands = "ugrizy"
            flux_name = self.config["flux_name"]
            for b in bands:
                try:
                    mag = nanojansky_to_mag_ab(group[f"{b}_{flux_name}"][:])
                    mag_err = nanojansky_err_to_mag_ab(
                        group[f"{b}_{flux_name}"][:], group[f"{b}_{flux_name}Err"][:]
                    )
                    group.create_dataset(f"mag_{b}", data=mag)
                    group.create_dataset(f"mag_err_{b}", data=mag_err)
                except KeyError:
                    warnings.warn(f"no flux {b}_{flux_name} in SSI GCR catalog")

            output_file.close()

class TXMatchSSI(PipelineStage):
    """
    Match an SSI injection catalog and a photometry catalog

    Default inputs are in TXPipe photometry catalog format

    Outpus a matched SSI catalog for further use
    """

    name = "TXMatchSSI"

    # TO DO: switch inputs from TXPipe format to GCR
    inputs = [
        ("injection_catalog", HDFFile),
        ("ssi_photometry_catalog", HDFFile),
    ]

    outputs = [
        ("matched_ssi_photometry_catalog", HDFFile),
    ]

    config_options = {
        "chunk_rows": 100000,
        "match_radius": 0.5,  # in arcseconds
        "magnification": 0,  # magnification label
    }

    def run(self):
        """
        Run the analysis for this stage.
        """

        # loop through chunks of the ssi photometry
        # catalog and match to the injections
        matched_cat = self.match_cats()

    def match_cats(self):
        """
        Match the injected catalogs with astropy tools

        TO DO: check if this step can be replaced with LSST probabalistic matching
        https://github.com/lsst/meas_astrom/blob/main/python/lsst/meas/astrom/match_probabilistic_task.py

        """
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        # prep the catalogs for reading
        # read ALL the ra and dec for the injection catalog
        inj_cat = self.open_input("injection_catalog")
        inj_coord = SkyCoord(
            ra=inj_cat["photometry/ra"][:] * u.degree,
            dec=inj_cat["photometry/dec"][:] * u.degree,
        )

        # loop over chunk of the photometry catalog
        phot_cat = self.open_input("ssi_photometry_catalog")
        nrows = phot_cat["photometry/ra"].shape[0]

        batch_size = self.config["chunk_rows"]
        n_chunk = int(np.ceil(nrows / batch_size))

        max_n = nrows
        match_outfile = self.setup_output(
            "matched_ssi_photometry_catalog",
            "photometry",
            inj_cat["photometry"],
            phot_cat["photometry"],
            max_n,
        )

        out_start = 0
        for ichunk, (in_start, in_end, data) in enumerate(
            self.iterate_hdf(
                "ssi_photometry_catalog", "photometry", ["ra", "dec"], batch_size
            )
        ):
            phot_coord = SkyCoord(
                ra=data["ra"] * u.degree,
                dec=data["dec"] * u.degree,
            )

            idx, d2d, d3d = phot_coord.match_to_catalog_sky(inj_coord)
            select_matches = d2d.value <= self.config["match_radius"] / 60.0 / 60.0
            nmatches = np.sum(select_matches)
            out_end = out_start + nmatches

            if nmatches != 0:
                print(out_start, out_end, nmatches)
                self.write_output(
                    match_outfile,
                    "photometry",
                    inj_cat["photometry"],
                    phot_cat["photometry"],
                    idx,
                    select_matches,
                    ichunk,
                    batch_size,
                    out_start,
                    out_end,
                )

            out_start = out_end

        self.finalize_output(match_outfile, "photometry", out_end)

    def setup_output(self, tag, group, inj_group, phot_group, n):
        """
        Prepare the hdf5 files
        """
        import h5py

        f = self.open_output(tag)
        g = f.create_group(group)

        for name, col in phot_group.items():
            g.create_dataset(name, shape=(n,), maxshape=n, dtype=col.dtype)
        for name, col in inj_group.items():
            g.create_dataset("inj_" + name, shape=(n,), maxshape=n, dtype=col.dtype)

        g.attrs["magnification"] = self.config["magnification"]

        # TO DO: add aditional metadata from inputs

        return f

    def write_output(
        self,
        outfile,
        group,
        inj_group,
        phot_group,
        idx,
        select_matches,
        ichunk,
        batch_size,
        start,
        end,
    ):
        """
        Write the matched catalog for a single chunk
        """
        g = outfile[group]
        for name, col in phot_group.items():
            g[name][start:end] = col[ichunk * batch_size : (ichunk + 1) * batch_size][
                select_matches
            ]
        for name, col in inj_group.items():
            g["inj_" + name][start:end] = col[idx][select_matches]

    def finalize_output(self, outfile, group, ntot):
        """
        Remove the excess rows,,c lose file
        """
        g = outfile[group]
        for name, col in g.items():
            col.resize((ntot,))
        outfile.close()
        return

class TXIngestSSIDESBalrog(TXIngestCatalogFits):
    """
    Base-stage for ingesting a DES SSI catalog AKA "Balrog"
    """

    name = "TXIngestSSIDESBalrog"

    def setup_output(self, output_name, column_names, dtypes, n):
        """
        For balrog, we need to include if statements to catch the 2D data entries
        and possibly add an extendedness column
        """
        cols = list(column_names.keys())
        output = self.open_output(output_name)
        g = output.create_group("photometry")
        for col in cols:
            dtype = dtypes[col]

            if dtype.subdtype is not None: #this is a multi-dimentional column
                assert dtype.subdtype[1]==(4,) #We are assuming this entry is a 2D array with 4 columns (corresponding to griz)
                dtype = dtype.subdtype[0]
                for b in "griz":
                    g.create_dataset(column_names[col] + f"_{b}", (n,), dtype=dtype)
            else:
                g.create_dataset(column_names[col], (n,), dtype=dtype)

            if col == "meas_EXTENDED_CLASS_SOF":
                #also create an "extendedness" column
                g.create_dataset("extendedness", (n,), dtype=dtype)
        
        return output, g

    def add_columns(self, g, input_name, column_names, chunk_rows, n):
        """
        For balrog, we need to include if statements to catch the 2D data entries
        and possibly add a extendedness column
        """
        cols = list(column_names.keys())
        for s, e, data in self.iterate_fits(input_name, 1, cols, chunk_rows):
            print(s, e, n)
            for col in cols:
                if len(data[col].shape) == 2:
                    assert data[col].shape[1] == 4
                    for iband, b in enumerate("griz"):
                        g[column_names[col] + f"_{b}"][s:e] = data[col][:, iband]
                else:
                    g[column_names[col]][s:e] = data[col]
                
                if col == "meas_EXTENDED_CLASS_SOF":
                    # "meas_EXTENDED_CLASS_SOF" is (0 or 1) for star, (2 or 3) for galaxy
                    # extendedness is 0 for star, 1 for galaxy
                    extendedness = np.where((data[col] == 2) | (data[col] == 3), 1, 0) 
                    g["extendedness"][s:e] = extendedness


class TXIngestSSIMatchedDESBalrog(TXIngestSSIDESBalrog):
    """
    Ingest a matched "SSI" catalog from DES (AKA Balrog)
    """

    name = "TXIngestSSIMatchedDESBalrog"

    inputs = [
        ("balrog_matched_catalog", FitsFile),
    ]

    outputs = [
        ("matched_ssi_photometry_catalog", HDFFile),
    ]

    def run(self):
        """
        Run the analysis for this stage.
        """
        print("Ingesting DES Balrog matched catalog")

        # we will only load a subset of columns to save space
        column_names = {
            "bal_id": "bal_id",  # Unique identifier for object (created during balrog process)
            "true_bdf_mag_deredden": "inj_mag",  # Magnitude of the original deep field object, dereddened
            "true_id": "inj_id",  # Original coadd_obj_id of deep field object
            "meas_id": "id",  # Coadd_object_id of injection
            "meas_ra": "ra",  # measured RA of the injection
            "meas_dec": "dec",  # measured DEC of the injection
            "meas_cm_mag_deredden": "mag",  # measured magnitude of the injection
            "meas_cm_max_flux_s2n": "snr", # measured S2N of the injection
            "meas_cm_T": "cm_T",  # measured size parameter T (x^2+y^2)
            "meas_EXTENDED_CLASS_SOF": "EXTENDED_CLASS_SOF",  # Star galaxy classifier (0,1=star, 2,3=Galaxy)
            "meas_FLAGS_GOLD_SOF_ONLY": "FLAGS_GOLD",  # Measured flags (short version)
        }
        dummy_columns = {
            "redshift_true": 10.0,
            "mag_err_g":-99.,
            "mag_err_r":-99.,
            "mag_err_i":-99.,
            "mag_err_z":-99.,
        }

        self.process_catalog(
            "balrog_matched_catalog",
            "matched_ssi_photometry_catalog",
            column_names,
            dummy_columns,
        )
    
class TXIngestSSIDetectionDESBalrog(TXIngestSSIDESBalrog):
    """
    Ingest an "SSI" "detection" catalog from DES (AKA Balrog)
    """

    name = "TXIngestSSIDetectionDESBalrog"

    inputs = [
        ("balrog_detection_catalog", FitsFile),
    ]

    outputs = [
        ("injection_catalog", HDFFile),
        ("ssi_detection_catalog", HDFFile),
    ]

    def run(self):
        """
        We will split the Balrog "detection" catalog into two catalogs
        One containing the injected catalog
        The other contains info on whether a given injection has been detected
        """

        ## Extract the injection catalog
        column_names_inj = {
            "bal_id": "bal_id",  # Unique identifier for object (created during balrog process)
            "true_ra": "ra",  # *injected* ra
            "true_dec": "dec",  # *injected* dec
            "true_bdf_mag_deredden": "inj_mag",  # true magnitude (de-reddened)
            "meas_FLAGS_GOLD_SOF_ONLY": "flags",  # measured data flags
            "meas_EXTENDED_CLASS_SOF": "meas_EXTENDED_CLASS_SOF",  # star galaxy separator
        }

        self.process_catalog(
            "balrog_detection_catalog", "injection_catalog", column_names_inj, {}
        )

        # Extract the "detection" file
        # We will only load a subset of columns to save space
        column_names_det = {
            "bal_id": "bal_id",  # Unique identifier for object (created during balrog process)
            "detected": "detected",  # 0 or 1. Is there a match between this object in injection_catalog and the ssi_photometry_catalog (search radius of 0.5'')
            "match_flag_0.5_asec": "match_flag_0.5_asec",  # 0,1 or 2. Is there a match between this object in injection_catalog and the ssi_uninjected_photometry_catalog (search radius 0.5''). 1 if detection is lower flux than injection, 2 if brighter
            "match_flag_0.75_asec": "match_flag_0.75_asec",  # as above but with search radius of 0.75''
            "match_flag_1.0_asec": "match_flag_1.0_asec",  # etc
            "match_flag_1.25_asec": "match_flag_1.25_asec",
            "match_flag_1.5_asec": "match_flag_1.5_asec",
            "match_flag_1.75_asec": "match_flag_1.75_asec",
            "match_flag_2.0_asec": "match_flag_2.0_asec",
        }

        self.process_catalog(
            "balrog_detection_catalog", "ssi_detection_catalog", column_names_det, {}
        )
