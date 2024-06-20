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

class TXIngestSSIGCR(PipelineStage):
    """
    Class for ingesting SSI catalogs using GCR

    Does not treat the injection or ssi photometry catalogs as formal inputs
    since they are not in a format TXPipe can recognize
    """

    name = "TXIngestSSIGCR"

    inputs = [
    ]

    outputs = [
        ("injection_catalog", HDFFile),
        ("ssi_photometry_catalog", HDFFile),
    ]

    config_options = {
        "injection_catalog_name":"",
        "ssi_photometry_catalog_name":"",
        "GCRcatalog_path":"",
        "all_cols":False,
        "magnification":0, # magnification label for run
    }

    def run(self):
        """
        Run the analysis for this stage.

        loads the catalogs using gcr and saves the relevent columns to a hdf5 format
        that TXPipe can read
        """
        if self.config["GCRcatalog_path"]!="":
            # This is needed to temporarily access the SSI runs on NERSC
            # As the final runs become more formalized, this could be removed  
            import sys
            sys.path.insert(0,self.config["GCRcatalog_path"])
        import GCRCatalogs

        #add loop over catalog types here
        output_catalogs = [
            "injection_catalog",
            "ssi_photometry_catalog",
        ]

        import ipdb
        ipdb.set_trace()

        for output_catalog_name in output_catalogs:

            catalog_name = self.config[f"{output_catalog_name}_name"]
            gc0 = GCRCatalogs.load_catalog(catalog_name)
            native_quantities = gc0.list_all_native_quantities()

            #Now translate all the relevent columns to the format TXPipe format
            #option 1, ingest all columns with existing names
            #option 2, ingest only a few relevant columns and give them TXPipe names
            output_file = self.open_output(output_catalog_name)
            group = output_file.create_group("photometry")
            
            if self.config['all_cols']:
                #find columns that contain non-nan data
                for q in native_quantities:
                    try:
                        qobj = gc0.get_quantities(q)
                    except KeyError:
                        warnings.warn(f"Skipping quantity {q}")
                        continue

                    try:
                        if np.isnan(qobj[q]).all():
                            continue #skip the quantities that are empty
                    except TypeError:
                        print(f'TypeError when checking for NaNs in {q}')

                    #TODO: add batch writing to hdf5 file
                    self.write_output(group, column_name, data)
                    group.create_dataset(q, data=qobj[q],  dtype=qobj[q].dtype)
            else:
                #save only the columns expected by the TXPipe photometry catalog
                #TO DO: do this
                pass

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

        #loop over chunk of the photometry catalog
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
