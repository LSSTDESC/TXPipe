from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, HDFFile, TextFile, QPBaseFile, ParquetFile
from ..utils import band_variants, metadetect_variants, Timer
import numpy as np
import glob
import math


class TXRubinIngest(PipelineStage):
    """
    This is a first pass at implementing an ingestion stage for the MetaDetect
    Catalog coming out of the Rubin pipeline. 

    Currently we expect Rubin to give us the data, one Tract at a time, so we will
    have to loop over the tracts to generate one catalog file.

    This first itteration will ingest HSC data processed by the Rubin pipeline.
    """

    name = "TXRubinIngest"
    parallel = False
    inputs = []

    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", QPBaseFile)
    ]

    config_options = {
        "pq_path": "",
        "tracts": "",
        "bands": "ugrizy",
        "input_tracts": True, #Assume that data is split into files per tract
        "input_variations": False, # Assume that the data is split into files per variation
        "moment": "wmom", #Which type of moment are we using for our files
    }

    def run(self):
        from pyarrow.parquet import ParquetFile
        import h5py
        from ..utils.hdf_tools import h5py_shorten, repack

        tracts = self.config["tracts"]
        pq_path = self.config['pq_path']
        bands = self.config['bands']

        cat_files = glob.glob(f"{pq_path}/INSERTNAME*.parq") #WE NEED THE NAME OF THE Tables

        tracts = [tract.strip() for tract in tracts.split(',')]
        print(f"Using {len(tracts)} tracts out of {len(cat_files)}")
        cat_files = [c for c in cat_files if c.split("/")[-1].split("_")[2] in tracts] #THIS PROBABLY WILL NEED TO CHANGE
        if len(cat_files) != len(tracts):
            raise ValueError("Some tracts not found")
        
        n=0
        for fn in cat_files:
            with ParquetFile(fn) as f:
                n += f.metadata.num_row # Might change might stay the same

        print(f"Full catalog size= {n:,}")


        # Defining the expected input columns, for photometry, and shape/shear!
        photo_cols = [ "id", "ra", "dec", "tract", "shear_type"]
        # List of possibly missing input: refExtendedness

        shape_cols = ["id", "ra", "dec", "tract","shear_type",
                      "psfrec_flags", "psfrec_g_1", "psfrec_g_2", "psfrec_T"
                      ]
        # The next set of columns depend on the moment used for the calculatialon
        moment=self.config["moment"]
        other_columns = [f"{moment}_psf_flags", f"{moment}_psf_g_1", f"{moment}_psf_g_2",
                         f"{moment}_psf_T", f"{moment}_obj_flags", f"{moment}_s2n",
                         f"{moment}_g_1", f"{moment}_g_2", f"{moment}_T",
                         f"{moment}_T_flags", f"{moment}_T_err", f"{moment}_T_ration"]
        
        # Magnitude columns, given to both photometry and shear catalogs
        for band in "ugrizy":
            for cols in [photo_cols, shape_cols]:
                cols.append(f"{moment}_band_flux_{band}")
                cols.append(f"{moment}_band_flux_err_{band}")
                cols.append(f"{moment}_band_flux_flags_{band}")

        cols = list(set(shape_cols+photo_cols+other_columns))

        nfile = len(cat_files)

        # making a list of what the columns that are in 
        self.shear_output_cols = metadetect_variants(
            "id",
            "tract",
            "g1",
            "g2",
            "T",
            "s2n",
            "T_err",
            "ra",
            "dec",
            "psf_g1",
            "psf_g2",
            "mcal_psf_g1",
            "mcal_psf_g2",
            "mcal_psf_T_mean",
            "weight",
        ) + band_variants(bands, "flux", "flux_err", "flux_flag", shear_catalog_type="metadetect")
        self.phot_output_cols = metadetect_variants(
            "id",
            "tract",
            "ra",
            "dec",
        ) + band_variants(bands,  "flux", "flux_err", "flux_flag", shear_catalog_type="metadetect")
        


        # Now using pyarrow to batch work through the files, and generate the TXPipe catalogs
        # THIS IS THE MAIN LOOP OF INGESTION
        batch_size = 65536  #default value
        for i, fn in enumerate(cat_files):
            with ParquetFile(fn) as f:
                n_chunk = math.ceil(f.metadata.num_rows/ batch_size)
                it = f.iter_batches(columns=cols)
                for j,d in enumerate(it):
                    d = {col.name: d[col.name].to_numpy(zero_copy_only=False) for col in d.schema}
                    if i == 0 and j == 0:
                        output_names = set(d.keys())
                        for col in cols:
                            assert col in output_names, f"Column {col} not found"
                    
                    photo_data = self.process_photometry_data(d)  # Function to run through the photometry part
                    shear_data = self.process_shear_data(d)  # Function to run through the shear/shape part

                    if i == 0 and j == 0:
                        photo_outfile = self.setup_output("photometry_catalog", "photometry", photo_data, n)
                        shear_outfile = self.setup_output("shear_catalog", "shear", shear_data, n)
                    else:
                        self.write_output(photo_outfile, "photometry", photo_data)
                        self.write_output(shear_outfile, "shear", shear_data)
                    print(f"Processing chunk {j+1}/{n_chunk} of file {i+1}/{nfile}")

        # UNSURE IF WE NEED THE THINGS BELOW HERE TO BEGIN WITH
        print("Trimming photometry columns:")
        for col in photo_data.keys():
            print("    ", col)
            h5py_shorten(photo_outfile["photometry"], col, end1)

        print("Trimming the shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, end2)
        
        photo_outfile.close()
        shear_outfile.close()

        #running h5repack on the files
        print("Repacking files")
        repack(self.get_output("photometry_catalog"))
        repack(self.get_output("shear_catalog"))


    def setup_output(self, tag, group, data):
        """
        setting up the output files, this is a joint method for both the 
        photometry and the shape catalogs.

        POTENTIAL ISSUE, if the first chunk doesn't include all variations, 
        NEED to implement stopgap/safety
        """
        import h5py 
        
        f = self.open_output(tag)
        g = f.create_group(group)
        g.attrs["bands"] = self.config['bands']
        g.attrs["moment"] = self.config['moment']

        for name, col in data.items:
            g.create_dataset(name, maxshape=(None,), data=col)

        return f
    
    def write_output(self, outfile, group, data):
        """
        writing output to the already made file!
        """
        g = outfile[group]
        for name, col in data.items():
            start = g[name].shape[0]
            n = len(col)

            g[name].resize((start+n,))

            g[name][start:start+n] = col
    
    def process_photometry_data(self, data):
        """
        Actually translating the photometry data into the output we need for TXPipe
        """
        bands = self.config['bands']
        moment = self.config['moment']
        output = {name: [] for name in self.phot_output_cols}
        for d in data:
            sheartype = d['shear_type']
            output[f"{sheartype}/ra"].append(d['ra'])
            output[f"{sheartype}/id"].append(d['id'])
            output[f"{sheartype}/dec"].append(d['dec'])
            output[f"{sheartype}/tract"].append(d["tract"])
            for band in bands:
                output[f"{sheartype}/flux_{band}"].append(d[f"{moment}_band_flux_{band}"])
                output[f"{sheartype}/flux_{band}_err"].append(d[f"{moment}_band_flux_err_{band}"])
                output[f"{sheartype}/flux_{band}_flag"].append(d[f"{moment}_band_flux_flags_{band}"])

        return output

    def process_shear_data(self, data):
        """
        Translating the input in to the need shape format for TXPipe
        """
        bands = self.config['bands']
        moment = self.config['moment']
        output = {name: [] for name in self.shear_output_cols}
        for d in data:
            sheartype = d['shear_type']
            output[f"{sheartype}/ra"].append(d['ra'])
            output[f"{sheartype}/id"].append(d['id'])
            output[f"{sheartype}/dec"].append(d['dec'])
            output[f"{sheartype}/tract"].append(d["tract"])
            output[f"{sheartype}/psf_flags"].append(d["{moment}_psf_flags"])
            output[f"{sheartype}/psf_g1"].append(d[f"{moment}_psf_g_1"])
            output[f"{sheartype}/psf_g2"].append(d[f"{moment}_psf_g_2"])
            output[f"{sheartype}/psf_T"].append(d[f"{moment}_psf_T"])
            output[f"{sheartype}/obj_flags"].append(d[f"{moment}_obj_flags"])
            output[f"{sheartype}/s2n"].append(d[f"{moment}_s2n"])
            output[f"{sheartype}/g1"].append(d[f"{moment}_g_1"])
            output[f"{sheartype}/g2"].append(d[f"{moment}_g_2"]) 
            output[f"{sheartype}/T"].append(d[f"{moment}_T"])
            output[f"{sheartype}/T_err"].append(d[f"{moment}_T_err"])
            output[f"{sheartype}/T_flags"].append(d[f"{moment}_T_flags"])
            output[f"{sheartype}/T_ration"].append(d[f"{moment}_T_ration"])
            for band in bands:
                output[f"{sheartype}/flux_{band}"].append(d[f"{moment}_band_flux_{band}"])
                output[f"{sheartype}/flux_err_{band}"].append(d[f"{moment}_band_flux_err_{band}"])
                output[f"{sheartype}/flux_flag_{band}"].append(d[f"{moment}_band_flux_flags_{band}"])

        return output