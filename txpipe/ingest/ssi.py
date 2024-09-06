from ..base_stage import PipelineStage
from ..data_types import (
    HDFFile,
)
from ..utils import (
    nanojansky_to_mag_ab,
    nanojansky_err_to_mag_ab,
)
import numpy as np
import warnings

# translate between GCR and TXpipe column names
column_names = {
    "coord_ra": "ra",
    "coord_dec": "dec",
}

class TXIngestSSIGCR(PipelineStage):
    """
    Class for ingesting SSI catalogs using GCR

    Does not treat the injection or ssi photometry catalogs as formal inputs
    since they are not in a format TXPipe can recognize
    """

    name = "TXIngestSSIGCR"

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

        #attempt to ingest three types of catalog
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
                    warnings.warn(f"quantity {q} was missing from the GCRCatalog object")
                    continue

                except TypeError:
                    warnings.warn(f"Quantity {q} coud not be saved as it has a data type not recognised by hdf5")

            # convert fluxes to mags using txpipe/utils/conversion.py
            bands = "ugrizy"
            flux_name = self.config["flux_name"]
            for b in bands:
                try:
                    mag = nanojansky_to_mag_ab(group[f"{b}_{flux_name}"][:])
                    mag_err = nanojansky_err_to_mag_ab(
                        group[f"{b}_{flux_name}"][:], group[f"{b}_{flux_name}Err"][:]
                    )
                    group.create_dataset(f"{b}_mag", data=mag)
                    group.create_dataset(f"{b}_mag_err", data=mag_err)
                except KeyError:
                    warnings.warn(f"no flux {b}_{flux_name} in SSI GCR catalog")

            output_file.close()


class TXMatchSSI(PipelineStage):
    """
    Class for ingesting SSI injection and photometry catalogs

    Default inputs are in TXPipe photometry catalog format

    Will perform its own matching between the catalogs to output a
    matched SSI catalog for further use

    TO DO: make a separate stage to ingest the matched catalog directly
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
        for ichunk, (in_start,in_end,data) in enumerate(self.iterate_hdf("ssi_photometry_catalog", "photometry", ["ra","dec"], batch_size)):
            phot_coord = SkyCoord(
                ra=data["ra"]*u.degree,
                dec=data["dec"]*u.degree,
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



class TXIngestSSIMatched(PipelineStage):
    """
    Base-stage for ingesting a matched SSI catalog

    This stage will just read in a file in a given format and output to a 
    HDF file that TXPIPE can use

    """

    name = "TXIngestSSIMatched"

    outputs = [
        ("matched_ssi_photometry_catalog", HDFFile),
    ]

    config_options = {
        "chunk_rows": 100_000,
        "magnification":0, # magnification label
    }


class TXIngestSSIMatchedDESBalrog(TXIngestSSIMatched):
    """
    Class for ingesting a matched "SSI" catalog from DES (AKA Balrog)
    """

    name = "TXIngestSSIMatchedDESBalrog"

    inputs = [
        ("balrog_matched_catalog", FitsFile),
    ]
  
    def run(self):
        """
        Run the analysis for this stage.
        """
        print('Ingesting DES Balrog matched catalog')

        #get some basic onfo about the input file
        f = self.open_input("balrog_matched_catalog")
        n = f[1].get_nrows()
        dtypes = f[1].get_rec_dtype()[0]
        f.close()

        print(f'{n} objects in matched catalog')

        chunk_rows = self.config["chunk_rows"]

        #we will only load a subset of columns to save space
        column_names = {
            "bal_id":                   "bal_id", 
            "true_bdf_mag_deredden":    "inj_mag", 
            "true_id":                  "inj_id", 
            "meas_id":                  "id", 
            "meas_ra":                  "ra", 
            "meas_dec":                 "dec", 
            "meas_cm_mag_deredden":     "mag",  
            "meas_cm_T":                "cm_T", 
            "meas_EXTENDED_CLASS_SOF":  "EXTENDED_CLASS_SOF",  
            "meas_FLAGS_GOLD_SOF_ONLY": "FLAGS_GOLD",
            }
        cols = list(column_names.keys())

        #set up the output file columns
        output = self.open_output("matched_ssi_photometry_catalog")
        g = output.create_group("photometry")
        for col in cols:
            dtype = dtypes[col]

            if "_mag" in col:
                #per band
                dtype = dtype.subdtype[0]
                for b in "griz":
                    g.create_dataset(column_names[col]+f"_{b}", (n,), dtype=dtype)
            else:
                g.create_dataset(column_names[col], (n,), dtype=dtype)

        #iterate over the input file and save to the output columns
        for (s, e, data) in self.iterate_fits("balrog_matched_catalog", 1, cols, chunk_rows):
            print(s,e,n)
            for col in cols:
                if "_mag" in col:
                    for iband,b in enumerate("griz"):
                        g[column_names[col]+f"_{b}"][s:e] = data[col][:,iband]
                else:
                    g[column_names[col]][s:e] = data[col]

        # Set up any dummy columns with sentinal values 
        # that were not in the original files
        dummy_columns = {
            "redshift_true":10.0,
            }
        for col_name in dummy_columns.keys():
            g.create_dataset(col_name, data=np.full(n,dummy_columns[col_name]))

        output.close()




