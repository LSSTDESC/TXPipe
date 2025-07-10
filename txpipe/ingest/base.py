from ..base_stage import PipelineStage
import numpy as np

class TXIngestCatalogBase(PipelineStage):
    """
    Base-Class for ingesting catalogs from external sources and saving to a format 
    TXPipe will understand
    """
    name = "TXIngestCatalogBase"

    def setup_output(self, output_name, column_names, dtypes, n):
        """
        Set up the output HDF5 file structure.

        Parameters
        ----------
        output_name : str
            The name of the output HDF5 file.

        column_names : dict
            dict of column names to include in the output file.
            Keys are output column names, and values are input column names.

        dtypes : dict
            A dictionary mapping the input catalog column names to their corresponding data types.

        n : int
            The total number of rows in the dataset.

        Returns
        -------
        tuple
            A tuple containing the open HDF5 output file object and the "photometry" group.
        """
        output = self.open_output(output_name)
        g = output.create_group("photometry")
        for new_col, old_col in column_names.items():
            dtype = dtypes[old_col]
            g.create_dataset(new_col, (n,), dtype=dtype)
        
        return output, g

    def process_catalog(self, input_name, output_name, column_names, dummy_columns):
        """
        Iterate over an input catalog, saving columns to a new h5 file

        Parameters
        ----------
        input_name : str
            label name of (FITS) input

        output_name : str
            label name of (HDF5) output

        column_names : dict
            A dictionary mapping the desired output column names to the input column names.
            Keys are output column names, and values are input column names.

        dummy_columns : dict
            A dictionary of columns to be added to the output with fixed values.
            Keys are the names of the dummy columns, and values are the constant values
            to fill those columns.

        """
        # get some basic info about the input file
        n, dtypes = self.get_meta(input_name)

        chunk_rows = self.config["chunk_rows"]

        # set up the output file columns 
        output, g = self.setup_output(output_name, column_names, dtypes, n)

        # iterate over the input file and save to the output columns
        self.add_columns(g, input_name, column_names, chunk_rows, n)

        # Set up any dummy columns with sentinal values
        # that were not in the original files
        for col_name, dummy_val in dummy_columns.keys():
            g.create_dataset(col_name, data=np.full(n, dummy_val))

        output.close()

class TXIngestCatalogFits(TXIngestCatalogBase):
    """
    Class for ingesting catalogs from FITS format and saving to a format 
    TXPipe will understand
    """
    name = "TXIngestCatalogFits"
    
    def get_meta(self, input_name, ):
        """
        Get some basic info about the input FITS file.

        Parameters
        ----------
        input_name : str
            The path to the input FITS file.

        Returns
        -------
        n : int
            Number of rows in the input data table.

        dtypes : numpy.dtype
            Numpy dtype object describing the data types of each column in the input.
        """
        f = self.open_input(input_name)
        n = f[1].get_nrows()
        dtypes = f[1].get_rec_dtype()[0]
        f.close()
        return n, dtypes

    def add_columns(self, g, input_name, column_names, chunk_rows, n):
        """
        Add data to the HDF5 output file in chunks.

        This method reads chunks of data from the input file and writes them
        to the corresponding datasets in the output file.

        Parameters
        ----------
        g : h5py.Group
            The HDF5 group where data will be written.

        input_name : str
            The name of the input file (e.g., a FITS file).

        column_names : dict
            Dict of column names to read from the input file.

        chunk_rows : int
            Number of rows to process in each chunk.

        n : int
            Total number of rows in the dataset.
        """
        cols = list(column_names.keys())
        input_cols = np.unique(list(column_names.values()))
        for s, e, data in self.iterate_fits(input_name, 1, input_cols, chunk_rows):
            print(s, e, n)
            for col in cols:
                g[col][s:e] = data[column_names[col]]

class TXIngestCatalogH5(TXIngestCatalogBase):
    """
    Class for ingesting catalogs from HDF5 files and saving to a format 
    TXPipe will understand
    """
    name = "TXIngestCatalogH5"
    
    def get_meta(self, input_name):
        """
        Get some basic info about the input file

        for h5 files I will assume the length of the longest dataset in the
        specified group is the number of objects in the catalog

        returns:
            n (int):
                number of rows in input data
            dtypes
                data types for each input column
        """
        import h5py 

        group = self.config['input_group_name']

        with self.open_input(input_name) as f:
            dtype_list = []
            n_list = []
            for name, item in f[group].items():
                if isinstance(item, h5py.Dataset):
                    dtype_list.append((name, item.dtype))
                    n_list.append(item.shape[0])
            assert len(n_list) > 0, f"No datasets found in group {group}"
        return np.max(n_list), np.dtype(dtype_list)

    def add_columns(self, g, input_name, column_names, chunk_rows, n):
        """
        Add data to the HDF5 output file in chunks.

        This method reads chunks of data from the input file and writes them
        to the corresponding datasets in the output file.

        Parameters
        ----------
        g : h5py.Group
            The HDF5 group where data will be written.

        input_name : str
            The name of the input file (e.g., a FITS file).

        column_names : dict
            Dict of column names to read from the input file.
            Keys are output column names, and values are input column names.

        chunk_rows : int
            Number of rows to process in each chunk.

        n : int
            Total number of rows in the dataset.
        """
        group = self.config['input_group_name']

        cols = list(column_names.keys())
        input_cols = np.unique(list(column_names.values()))
        for s, e, data in self.iterate_hdf(input_name, group, input_cols, chunk_rows):
            print(s, e, n)
            for col in cols:
                g[col][s:e] = data[column_names[col]]

class TXIngestMapsBase(PipelineStage):
    """
    Base-Class for ingesting maps from external sources and saving to a format 
    TXPipe will understand
    """
    name = "TXIngestMapsBase"

    config_options = {
        "input_nside": int,
        "input_nest": True,
    }

    def setup_output(self, output_name, groups):
        """
        Set up the output HDF5 file structure.

        Parameters
        ----------
        output_name : str
            The name of the output HDF5 file.

        groups : list of str
            names of the ingested maps
        """

        output = self.open_output(output_name)
        maps_group = output.create_group("maps")

        for group_name in groups:
            maps_group.create_group(group_name)
        
        return output

    def process_maps(self, input_filepaths, input_labels, output_name):
        """
        Add some input maps to an HDF5 file

        Parameters
        ----------
        input_filepaths : list of str
            List of input healsparse map files

        input_labels : list of str
            List of labels for each map. These will be the group names in the output file
            Same ordering as input_filepaths
        
        output_name : str
            The name of the output HDF5 file.
        """
        #output = self.setup_output(output_name, input_labels)

        maps = {}

        for input_label, input_file in zip(input_labels, input_filepaths):
            print(f'Processing map {input_label}')
            pixel, value = self.load_map(input_file, return_nest=False)
            maps[input_label] = (pixel, value)

        metadata = {
            "pixelization":"healpix",
            "nside":self.config["input_nside"],
            "nest": False, #currently TXPipe defaults to ring format
        }
        print(f"Input nside {self.config['input_nside']}")

        # Write the output maps to the output file
        with self.open_output(output_name, wrapper=True) as out:
            for map_name, (pix, m) in maps.items():
                out.write_map(map_name, pix, m, metadata)
            out.file['maps'].attrs.update(metadata)

class TXIngestMapsHsp(TXIngestMapsBase):
    """
    Class for ingesting maps from external healsparse files and saving to a format 
    TXPipe will understand
    """
    name = "TXIngestMapsHsp"

    def load_map(self, input_filepath, return_nest=False):
        """
        Add a single map to the HDF5 file 

        Parameters
        ----------
        input_filepath : str
            input healsparse map files

        return_nest: bool
            if True output will be in nest format
        """
        import healpy as hp
        import healsparse as hsp

        hsmap = hsp.HealSparseMap.read(input_filepath)
        nside = hsmap.nside_sparse
        valid_pix = hsmap.valid_pixels
        if return_nest:
            return valid_pix, hsmap[valid_pix]
        else:
            return hp.nest2ring(nside, valid_pix), hsmap[valid_pix]
