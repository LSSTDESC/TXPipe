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
            dtype = dtypes[col]
            g.create_dataset(column_names[col], (n,), dtype=dtype)
        
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
            A dictionary mapping the input column names to the desired output column names.
            Keys are input column names, and values are the corresponding output names.

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

class TXIngestCatalogFits(PipelineStage):
    """
    Base-Class for ingesting catalogs from external sources and saving to a format 
    TXPipe will understand
    """
    name = "TXIngestCatalogFits"
    
    def get_meta(self, input_name):
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
        for s, e, data in self.iterate_fits(input_name, 1, cols, chunk_rows):
            print(s, e, n)
            for col in cols:
                g[column_names[col]][s:e] = data[col]