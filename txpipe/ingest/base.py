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

        dtypes : dict
            A dictionary mapping column names to their corresponding data types.

        n : int
            The total number of rows in the dataset.

        Returns
        -------
        tuple
            A tuple containing the open HDF5 output file object and the "photometry" group.
        """
        cols = list(column_names.keys())
        output = self.open_output(output_name)
        g = output.create_group("photometry")
        for col in cols:
            dtype = dtypes[column_names[col]]
            g.create_dataset(col, (n,), dtype=dtype)
        
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
        for col_name in dummy_columns.keys():
            g.create_dataset(col_name, data=np.full(n, dummy_columns[col_name]))

        output.close()

class TXIngestCatalogFits(TXIngestCatalogBase):
    """
    Class for ingesting catalogs from FITS format and saving to a format 
    TXPipe will understand
    """
    name = "TXIngestCatalogFits"
    
    def get_meta(self, input_name, ):
        """
        Get some basic info about the input file

        returns:
            n (int):
                number of rows in input data
            dtypes
                data types for each input column
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