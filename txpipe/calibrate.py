from .base_stage import PipelineStage
from .data_types import ShearCatalog, TomographyCatalog
from .utils import read_shear_catalog_type, Calibrator, Splitter, SourceNumberDensityStats
import numpy as np


class TXShearCalibration(PipelineStage):
    """Split the shear catalog into calibrated bins suitable for 2pt analysis."""

    name = "TXShearCalibration"
    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("calibrated_shear_catalog", ShearCatalog),
    ]

    config_options = {
        "use_true_shear": False,
        "chunk_rows": 100_000,
        "subtract_mean_shear": True,
    }

    def run(self):

        #  Extract the configuration parameters
        cat_type = read_shear_catalog_type(self)
        use_true = self.config["use_true_shear"]
        subtract_mean_shear = self.config["subtract_mean_shear"]

        # Prepare the output file, and
        output_file, splitter, nbin = self.setup_output()

        #  Load the calibrators.  If using the true shear no calibration
        # is needed
        tomo_file = self.get_input("shear_tomography_catalog")
        cals, cal2d = Calibrator.load(tomo_file, null=use_true)

        # These are always named the same
        cat_cols = ["ra", "dec", "weight"]
        # The catalog columns are named differently in different cases
        #  Get the correct shear catalogs
        if use_true:
            cat_cols += ["true_g1", "true_g2"]
        elif cat_type == "metacal":
            cat_cols += ["mcal_g1", "mcal_g2"]
        else:
            cat_cols += ["g1", "g2"]

        output_cols = ["ra", "dec", "g1", "g2", "weight"]

        # We parallelize by bin.  This isn't ideal but we don't know the number
        # of objects in each bin per chunk, so we can't parallelize in full.  This
        #  is a quick stage though.
        bins = list(range(nbin)) + ["all"]
        my_bins = list(self.split_tasks_by_rank(bins))

        # Print out which bins this proc will do, also as a prompt to the user
        #  in case they're wondering why adding procs doesn't help
        if my_bins:
            my_bins_text = ", ".join(str(x) for x in my_bins)
            print(f"Process {self.rank} collating bins: [{my_bins_text}]")
        else:
            print(f"Note: Process {self.rank} will not do anything.")

        # make the iterator that loops through data
        it = self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            "shear_catalog",
            "shear",
            cat_cols,
            "shear_tomography_catalog",
            "tomography",
            ["source_bin"],
            parallel=False,
        )

        #  Main loop
        for s, e, data in it:
            # Rename mcal_g1 -> g1 etc
            self.rename_metacal(data)

            #  Now output the calibrated bin data for this processor
            for b in my_bins:

                # Select objects to go in this bin
                if b == "all":
                    # the 2D case is any object from any other bin
                    w = np.where(data["source_bin"] >= 0)
                    cal = cal2d
                else:
                    # otherwise just objects in this bin
                    w = np.where(data["source_bin"] == b)
                    cal = cals[b]

                # Cut down the data to just this selection for output
                d = {name: data[name][w] for name in output_cols}

                # Calibrate the shear columns
                d["g1"], d["g2"] = cal.apply(d["g1"], d["g2"], subtract_mean=subtract_mean_shear)

                # Write output, keeping track of sizes
                splitter.write_bin(d, b)

        splitter.finish(my_bins)
        output_file.close()

    def setup_output(self):
        # count the expected number of objects per bin from the tomo data
        with self.open_input("shear_tomography_catalog") as f:
            counts = f["tomography/source_counts"][:]
            count2d = f["tomography/source_counts_2d"][0]
            nbin = len(counts)

        # Prepare the calibrated output catalog
        f = self.open_output("calibrated_shear_catalog", parallel=True)

        #  we only retain these columns
        cols = ["ra", "dec", "weight", "g1", "g2"]

        # structure is /shear/bin_1, /shear/bin_2, etc
        g = f.create_group("shear")
        g.attrs["nbin"] = nbin
        g.attrs["nbin_source"] = nbin

        bins = {b: c for b, c in enumerate(counts)}
        bins["all"] = count2d
        splitter = Splitter(g, "bin", cols, bins)

        return f, splitter, nbin

    def rename_metacal(self, d):
        #  rename the columns so they're always just g1, g2.
        #  First determine the renaming we should do
        if "true_g1" in d:
            prefix = "true"
        elif "mcal_g1" in d:
            prefix = "mcal"
        else:
            return

        #  Then rename
        d["g1"] = d[f"{prefix}_g1"]
        d["g2"] = d[f"{prefix}_g2"]
        del d[f"{prefix}_g1"], d[f"{prefix}_g2"]
