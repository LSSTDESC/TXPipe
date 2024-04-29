from ..base_stage import PipelineStage
from ..data_types import (
    YamlFile,
    HDFFile,
    FitsFile,
)
from ..utils import LensNumberDensityStats, Splitter, rename_iterated
from ..binning import build_tomographic_classifier, apply_classifier
import numpy as np
import warnings


class TXIngestSSI(PipelineStage):
    """
    Class for ingesting SSI injection and photometry catalogs
    
    Will perform its own matching between the catalogs to output a 
    matched SSI catalog for further use

    TO DO: make a separate stage to ingest the matched catalog directly
    """

    name = "TXIngestSSI"

    # TO DO: switch inputs from TXPipe format to either GCR or butler 
    inputs = [
        ("injection_catalog", HDFFile),
        ("ssi_photometry_catalog", HDFFile),
    ]

    outputs = [
        ("matched_ssi_photometry_catalog", HDFFile),
    ]

    config_options = {
        "chunk_rows": 100000,
        "match_radius": 0.5, # in arcseconds
        "magnification":0, # magnification label
    }

    def run(self):
        """
        Run the analysis for this stage.
        """

        # loop through chunks of the ssi photometry
        # catalog and match to the injections
        matched_cat = self.match_cats()

        # output the matched catalog with as much SSI metadata
        # stored as possible 

    def match_cats(self):
        """
        Match the injected catalogs with astropy tools

        TO DO: check if this step can be replaced with LSST probabalistic matching
        https://github.com/lsst/meas_astrom/blob/main/python/lsst/meas/astrom/match_probabilistic_task.py
        
        """
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        # prep the catalogs for reading
        #read ALL the ra and dec for the injection catalog
        inj_cat  = self.open_input("injection_catalog")
        inj_coord = SkyCoord(
            ra  = inj_cat['photometry/ra'][:]*u.degree, 
            dec = inj_cat['photometry/dec'][:]*u.degree
            )

        #loop over chunkc of the photometry catalog
        phot_cat = self.open_input("ssi_photometry_catalog")
        nrows = phot_cat['photometry/ra'].shape[0]

        batch_size = self.config["chunk_rows"]
        n_chunk = int(np.ceil( nrows / batch_size))

        max_n = nrows
        match_outfile = self.setup_output("matched_ssi_photometry_catalog", "photometry", inj_cat['photometry'], phot_cat['photometry'], max_n)

        start1 = 0
        for ichunk in range(n_chunk):
            phot_coord = SkyCoord(
                ra  = phot_cat['photometry/ra'][ichunk*batch_size:(ichunk+1)*batch_size]*u.degree, 
                dec = phot_cat['photometry/dec'][ichunk*batch_size:(ichunk+1)*batch_size]*u.degree
                )

            idx, d2d, d3d = phot_coord.match_to_catalog_sky(inj_coord)
            select_matches = (d2d.value <= self.config["match_radius"]/60./60.)
            nmatches = np.sum(select_matches)
            end1 = start1 + nmatches

            if nmatches != 0 :
                print(start1, end1, nmatches)
                self.write_output(
                    match_outfile, "photometry", inj_cat['photometry'], phot_cat['photometry'], 
                    idx, select_matches, ichunk, batch_size, start1, end1)

            start1 = end1

        self.finalize_output(match_outfile, "photometry", end1)


    def setup_output(self, tag, group, inj_group, phot_group, n):
        """
        Prepare the hdf5 files
        """
        import h5py

        f = self.open_output(tag)
        g = f.create_group(group)

        for name, col in phot_group.items():
            g.create_dataset(name, shape=(n,),  maxshape=n, dtype=col.dtype)
        for name, col in inj_group.items():
            g.create_dataset("inj_"+name, shape=(n,), maxshape=n, dtype=col.dtype)        

        g.attrs['magnification'] = self.config['magnification']

        #TO DO: add aditional metadata from inputs

        return f

    def write_output(self, outfile, group, inj_group, phot_group, idx, select_matches, ichunk, batch_size, start, end):
        """
        Write the matched catalog for a single chunk
        """
        g = outfile[group]
        for name, col in phot_group.items():
            g[name][start:end] = col[ichunk*batch_size:(ichunk+1)*batch_size][select_matches]
        for name, col in inj_group.items():
            g["inj_"+name][start:end] = col[idx][select_matches]

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












