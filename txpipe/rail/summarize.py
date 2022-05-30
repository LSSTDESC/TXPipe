from ..base_stage import PipelineStage
from ..data_types import ParquetFile, HDFFile, PickleFile, QPFile


class TXParqetToHDF(PipelineStage):
    """Generic stage to convert a Parquet File to HDF

    This stage uses Eric Charles' alias feature.  In the config
    you set:
    TXParqetToHDF:
        aliases:
            input: actual_input_tag
            output: actual_output_tag

    and the appropriate tags will be replaced.
    """
    name = "TXParqetToHDF"
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


class PZRailSummarizeLens(PipelineStage):
    name = "PZRailSummarizeLens"
    inputs = [
        ("lens_tomography_catalog", HDFFile),
        ("photometry_catalog", HDFFile),
        ("nz_dir_model", PickleFile),

    ]
    outputs = [
        ("lens_photoz_stack", HDFFile),
    ]

    # pull these out automatically
    config_options = {
        "leafsize": 20,

    }

    def run(self):
        import rail.estimation
        import pickle
        from rail.estimation.algos.NZDir import NZDir
        from rail.core import DataStore, TableHandle

        model_filename = self.get_input("nz_dir_model")


        # The usual way of opening pickle files puts a load
        # of provenance at the start of them. External pickle files
        # like the ones from RAIL don't have this, so the file content
        # comes out wrong.
        # Once we've moved the provenance stuff from TXPipe to ceci
        # then prov tracking should be harmonized, then we can replace
        # these two lines:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        # with these:
        # with self.open_input("nz_dir_model", wrapper=True) as f:
        #     model = f.read()


        bands = model["szusecols"]


        # This is the bit that will not work with realistically sized
        # data sets. Need the RAIL parallel interface when ready.
        with self.open_input("photometry_catalog") as f:
            g = f['photometry']
            full_data = {b: g[f"/photometry/mag_{b}"][:] for b in bands}


        with self.open_input("lens_tomography_catalog") as f:
            g = f['tomography']
            nbin = g.attrs['nbin_lens']
            bins = g['lens_bin'][:]

        # Just do things with the first bin to begin with
        qp_per_bin = []
        for i in range(nbin):

            index = (bins==i)
            nobj = index.sum()
            print(f"Computing n(z) for bin {i}: {nobj} objects")
            data = {b: full_data[b][index] for b in bands}

            # any reason not to make this manually?
            # This seems like it would needlessly store lots of data
            data_handle = NZDir.data_store.add_data(f"lens_bin_{i}", data, TableHandle)

            sub_config = {
                # TODO: Move to config
                "leafsize": 20,
                "zmin": 0.0,
                "zmax": 3.0,
                "nzbins": 50,
                "model": model,  # not sure if I can put the model in here. Try.
                "usecols": bands,
                "hdf5_groupname": "",
                "phot_weightcol":"",
            }

            sub_stage = NZDir.make_stage(name=f"NZDir_{i}", **sub_config)
            bin_qp = sub_stage.estimate(data_handle)
            qp_per_bin.append(bin_qp.data)


        for q in qp_per_bin[1:]:
            qp_per_bin[0].append(q)

        # TODO: Metadata
        qp_per_bin[0].write_to(self.get_output("lens_photoz_stack"))
