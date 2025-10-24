from ..base_stage import PipelineStage
from ..data_types import HDFFile, PickleFile, PNGFile, QPMultiFile, QPNOfZFile, BinnedCatalog, TomographyCatalog
import numpy as np
import os
from ceci.config import StageParameter

class PZRailSummarizeBase(PipelineStage):
    """
    Base class to build the n(z) from some tomographic bins and 
    collate the results from all the bins into a single file
    
    "CatSummarizer" expects a catalog input (e.g. magnitudes)
    "PZSummarizer" expects a photo-z input (e.g. point estimates or pdfs)
    """
    name = "PZRailSummarizeBase"
    outputs = [
        ("photoz_stack", QPNOfZFile),
        ("photoz_realizations", QPMultiFile),
    ]

    def run(self):
        import rail.estimation
        import pickle
        from rail.core.data import TableHandle
        from rail.core.data import DataStore
        import qp
        import importlib
        summarize_module = importlib.import_module(self.config["module"])


        # Get the list of bins from the input
        # Mostly these are numbered, bin_0, bin_1, etc.
        # But there is also often a special bin_all
        # that is non-tomographic
        bins = self.get_bin_list()

        # These parameters are common to all the bins runs
        if "binned_catalog" in self.input_tags():
            input_name = self.get_input("binned_catalog")
        elif "photoz_pdfs" in self.input_tags():
            input_name = self.get_input("photoz_pdfs")
        else:
            raise RuntimeError('class inputs should contain either binned_catalog or photoz_pdfs')
        
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
            "input": input_name,
            "output_mode": "default", #only some summarizer classes need this it seems
        }

        # This is the requested RAIL Summarizer class
        substage_class = getattr(summarize_module, self.config["summarizer"])

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

            extra_sub_config = self.get_extra_sub_config(b)

            # Create the sub-stage.
            run_nzdir = substage_class.make_stage(output=realizations_output,
                                                  single_NZ=stack_output,
                                                  **sub_config,
                                                  **extra_sub_config,
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
        

class PZRailSummarize(PZRailSummarizeBase):
    """
    Runs a specified RAIL "CatSummarizer" on each tomographic bin and 
    collate the results from all the bins into a single file
    
    "CatSummarizer" expects a catalog input, rather than a PZ input
    """
    name = "PZRailSummarize"
    inputs = [
        ("binned_catalog", BinnedCatalog),
        ("model", PickleFile),
    ]

    # pull these out automatically
    config_options = {
        "catalog_group": StageParameter(str, '', msg="Group name in the catalog file for tomographic bins."),
        "mag_prefix": StageParameter(str, "photometry/mag_", msg="Prefix for magnitude columns in the catalog."),
        "tomography_name": StageParameter(str, '', msg="Name of the tomography scheme."),
        "bands": StageParameter(str, "ugrizy", msg="Bands to use for summarization."),
        "summarizer": StageParameter(str, "NZDirSummarizer", msg="Name of the RAIL summarizer class to use."),
        "module": StageParameter(str, "rail.estimation.algos.nz_dir", msg="Python module path for the summarizer class."),
    }

    def get_bin_list(self):
        """
        Returns a list of bins in the binned catalog file
        """
        group_name = self.config["catalog_group"]
        with self.open_input("binned_catalog", wrapper=True) as f:
            bins = f.get_bins(group_name)
        return bins

    def get_extra_sub_config(self, bin):
        """
        Additional config options needed by the CatSummarizers
        """
        group_name = self.config["catalog_group"]
        return {"hdf5_groupname": f"/{group_name}/{bin}"}

class PZRailPZSummarize(PZRailSummarizeBase):
    """
    Runs a specified RAIL *masked* "PZSummarizer" on each tomographic bin and 
    collate the results from all the bins into a single file
    
    "PZSummarizer" expects a photoz pdf input, rather than a catalog
    """
    name = "PZRailPZSummarize"
    inputs = [
        ("photoz_pdfs", HDFFile),
        ("tomography_catalog", TomographyCatalog),
        ("model", PickleFile),
    ]

    # pull these out automatically
    config_options = {
        "catalog_group": StageParameter(str, '', msg="Group name in the catalog file for tomographic bins."),
        "mag_prefix": StageParameter(str, "photometry/mag_", msg="Prefix for magnitude columns in the catalog."),
        "tomography_name": StageParameter(str, '', msg="Name of the tomography scheme."),
        "bands": StageParameter(str, "ugrizy", msg="Bands to use for summarization."),
        "summarizer": StageParameter(str, "PointEstHistMaskedSummarizer", msg="Name of the RAIL summarizer class to use."),
        "module": StageParameter(str, "rail.estimation.algos.point_est_hist", msg="Python module path for the summarizer class."),
    }

    def get_bin_list(self):
        """
        Returns a list of bins in the binned catalog file
        """
        group_name = self.config['hdf5_groupname']
        bins = []
        with self.open_input("tomography_catalog") as f:
            unique_bins = np.unique(f[f'{group_name}/class_id'][:])
            bins =  [f"bin_{bin_index}" for bin_index in unique_bins[unique_bins != -1]] #do not include the "unselected" bin, -1
        return bins

    def get_extra_sub_config(self, bin):
        """
        Additional config options needed by the PZSummarizers
        """
        with self.open_input("tomography_catalog",wrapper=True) as f:
            tomo_path = f.path
        bin_index = int(bin.split('_')[-1])
        
        extra_sub_config = {
            "selected_bin": bin_index, 
            "tomography_bins":tomo_path, 
            "hdf5_groupname":self.config['hdf5_groupname'],
            }      
        print(extra_sub_config) 
        return extra_sub_config


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
        "zmax": StageParameter(float, 3.0, msg="Maximum redshift for plotting."),
        "nz": StageParameter(int, 301, msg="Number of redshift bins for plotting."),
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

