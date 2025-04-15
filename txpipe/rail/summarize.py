from ..base_stage import PipelineStage
from ..data_types import HDFFile, PickleFile, PNGFile, QPMultiFile, QPNOfZFile, BinnedCatalog
import numpy as np
import collections
import os

class PZRailSummarize(PipelineStage):
    name = "PZRailSummarize"
    inputs = [
        ("binned_catalog", BinnedCatalog),
        ("model", PickleFile),

    ]
    outputs = [
        ("photoz_stack", QPNOfZFile),
        ("photoz_realizations", QPMultiFile),
    ]

    # pull these out automatically
    config_options = {
        "catalog_group": str,
        "mag_prefix": "photometry/mag_",
        "tomography_name": str,
        "bands": "ugrizy",
    }

    def run(self):
        import rail.estimation
        import pickle
        from rail.estimation.algos.nz_dir import NZDirSummarizer
        from rail.core.data import TableHandle
        from rail.core.data import DataStore
        import qp


        # Get the list of bins from the binned catalog.
        # Mostly these are numbered, bin_0, bin_1, etc.
        # But there is also often a special bin_all
        # that is non-tomographic
        group_name = self.config["catalog_group"]
        with self.open_input("binned_catalog", wrapper=True) as f:
            bins = f.get_bins(group_name)

        # These parameters are common to all the bins runs
        cat_name = self.get_input("binned_catalog")
        model = self.get_input("model")

        # We will run the ensemble estimation for each bin, so we want a separate output file for each
        # we use the main file name as the base name for this, but add the bin name to the end.
        # So to do this we split up the file name into root and extension, and then add the bin name
        # to the root.
        out1_root, out1_ext = os.path.splitext(self.get_output("photoz_realizations", final_name=True) )
        out2_root, out2_ext = os.path.splitext(self.get_output("photoz_stack", final_name=True) )


        # Configuration options that will be needed by the sub-stage
        sub_config = {
            "model": model,
            "usecols": [f"mag_{b}" for b in self.config["bands"]],
            "phot_weightcol":"weight",
            "output_mode": "none",
            "comm": self.comm,
            "input": cat_name,
        }

        # We want to make this customizable.
        substage_class = NZDirSummarizer

        # Move relevant configuration items from the config
        # for this stage into the substage config
        for k, v in self.config.items():
            if k in substage_class.config_options:
                sub_config[k] = v

        # We collate ensemble results for both a single primary n(z) estimate,
        # and for a suite of realizations, both for each bin.
        main_ensemble_results = []
        realizations_results = []

        for b in bins:
            print("Running ensemble estimation for bin: ", b)
            
            # Make the file names for this bin
            realizations_output = "{}_{}{}".format(out1_root, b, out1_ext)
            stack_output = "{}_{}{}".format(out2_root, b, out2_ext)

            # Create the sub-stage.
            run_nzdir = substage_class.make_stage(hdf5_groupname=f"/{group_name}/{b}",
                                                  output=realizations_output,
                                                  single_NZ=stack_output,
                                                  **sub_config
                                                )
            # I have never really understood the RAIL DataStore.
            # But if I don't do this the system complains that there is already an
            # output with the name we have given when we get to the second iteration
            ds = DataStore()
            run_nzdir.data_store = ds

            # actually run and finalize the stage
            run_nzdir.run()
            run_nzdir.finalize()

            if self.comm is not None:
                self.comm.Barrier()

            # read the result. This is small so can stay in memory.
            main_ensemble_results.append(qp.read(stack_output))
            realizations_results.append(qp.read(realizations_output))

            
        # Only the root process writes the data, so the others are done now.
        if self.rank != 0:
            return

        # Collate the results from all the bins into a single file
        # and write to the main output file.
        combined_qp = qp.concatenate(main_ensemble_results)
        with self.open_output("photoz_stack", wrapper=True) as f:
            f.write_ensemble(combined_qp)

        # Write the realizations to the its file, again, collating all the bins
        with self.open_output("photoz_realizations", wrapper=True) as f:
            for b, realizations in zip(bins, realizations_results):
                f.write_ensemble(realizations, b)
        



                
class PZRealizationsPlot(PipelineStage):
    name = "PZRealizationsPlot"
    parallel = False

    inputs = [
        ("photoz_realizations", QPMultiFile),
    ]

    outputs = [
        ("photoz_realizations_plot", PNGFile),
    ]

    config_options = {
        "zmax": 3.0,
        "nz": 301,
    }

    def run(self):
        import h5py
        import matplotlib.pyplot as plt


        zmax = self.config["zmax"]
        nz = self.config["nz"]
        z = np.linspace(0, zmax, nz)

        with self.open_input("photoz_realizations", wrapper=True) as f:
            names = f.get_names()
            pdfs = {}
            for tomo_bin in names:
                realizations = f.read_ensemble(tomo_bin)
                pdfs[tomo_bin] = realizations.pdf(z)

        # Here nbin includes the 2D bin
        nbin = len(names)
        with self.open_output("photoz_realizations_plot", wrapper=True, figsize=(6, 4*nbin)) as fig:
            axes = fig.file.subplots(len(names), 1)
            for i, tomo_bin in enumerate(names):
                ax = axes[i]
                pdfs_i = pdfs[tomo_bin]
                print(pdfs_i.shape)
                ax.plot(z, pdfs_i.mean(0))
                ax.fill_between(z, pdfs_i.min(0), pdfs_i.max(0), alpha=0.2)

                ax.set_xlabel("z")
                ax.set_ylabel(f"{tomo_bin} n(z)")

