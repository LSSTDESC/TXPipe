from ..base_stage import PipelineStage
from ..data_types import ParquetFile, HDFFile, FitsFile

class TXParqetToHDF(PipelineStage):
    """Generic stage to convert a Parquet File to HDF

    This will need to use aliases to be any use.
    """
    name = "TXParqetToHDF"
    parallel = False
    inputs = [
        ("input", ParquetFile),
    ]
    outputs = [
        ("output", HDFFile),
    ]
    config_options = {
        "hdf_group": "/"
    }

    def run(self):
        import pyarrow.parquet
        input_tag = self.get_aliased_tag("input")
        output = self.open_output("output")
        group = self.config['hdf_group']
        if group == "/":
            out_group = output
        else:
            out_group = output.create_group(group)


        # Get the column names.  Oddly, doing f.schema below
        # gives you a different kind of object without the types included

        # Currently ceci is inconsistent about self.get_input vs self.open_input
        # and also self.
        schema = pyarrow.parquet.read_schema(self.get_input(input_tag))

        input_ = self.open_input("input", wrapper=False)
        n = input_.metadata.num_rows

        # Create the columns in the output file
        for name, dtype in zip(schema.names, schema.types):
            dtype = dtype.to_pandas_dtype()
            out_group.create_dataset(name, shape=(n,), dtype=dtype)

        # Copy the data across in batches
        s = 0
        for batch in input_.iter_batches():
            e = s + batch.num_rows
            for name in schema.names:
                out_group[name][s:e] = batch[name]
            s = e

        # There seems to be no close method on parquet files
        output.close()
