from .base_stage import PipelineStage
from .data_types import ShearCatalog, TomographyCatalog
from .utils import read_shear_catalog_type, Calibrator, Splitter, rename_iterated
import numpy as np


class TXShearCalibration(PipelineStage):
    """Split the shear catalog into calibrated bins

    This class runs after source selection has been done, because the final
    calibration factor can only be estimated once we have read the entire catalog
    and chosen tomographic bins (since it is an ensemble average of cal factors).

    Once that stage has run and computed both the tomographic bin for each sample
    and the calibration factors, this stage takes the full catalog and splits it
    into one HDF5 group per bin.  This has several advantages:
    - all the calibration can happen in one place rather than
      differently in real space and Fourier
    - we can load just the galaxies we want for a single bin in later TreeCorr
      stages, rather than loading the full catalog and then splitting and calibrating
      it.
    - the low_mem option in TreeCorr can be used because the catalogs are on disc
      and contiguous.
    - it opens up other memory saving options planned for TreeCorr.

    We are not (yet) saving per-patch catalogs for TreeCorr here. We might want to
    do that later.
    """

    name = "TXShearCalibration"
    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("binned_shear_catalog", ShearCatalog),
    ]

    config_options = {
        "use_true_shear": False,
        "chunk_rows": 100_000,
        "subtract_mean_shear": True,
        "extra_cols": [""],
    }

    def run(self):

        #  Extract the configuration parameters
        cat_type = read_shear_catalog_type(self)
        use_true = self.config["use_true_shear"]
        extra_cols = [c for c in self.config["extra_cols"] if c]
        subtract_mean_shear = self.config["subtract_mean_shear"]

        # Prepare the output file, and create a splitter object,
        # whose job is to save the separate bins to separate HDF5
        # extensions depending on the tomographic bin
        output_file, splitter, nbin = self.setup_output(extra_cols)

        #  Load the calibrators.  If using the true shear no calibration
        # is needed
        tomo_file = self.get_input("shear_tomography_catalog")
        cals, cal2d = Calibrator.load(tomo_file, null=use_true)

        # The catalog columns are named differently in different cases
        #  Get the correct shear catalogs
        with self.open_input("shear_catalog", wrapper=True) as f:
            cat_cols, renames = f.get_primary_catalog_names()

            cat_cols += [f"00/{c}" for c in extra_cols]
            renames.update({f"00/{c}":c for c in extra_cols})
    
        if cat_type!='hsc':
            output_cols = ["ra", "dec", "weight", "g1", "g2"] + extra_cols
        else:
            output_cols = ["ra", "dec", "weight", "g1", "g2","c1","c2"] + extra_cols

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
            ["bin"],
            parallel=False,
        )

        #  Main loop
        for s, e, data in rename_iterated(it, renames):

            if self.rank == 0:
                print(f"Rank 0 processing data {s:,} - {e:,}")

            # Rename mcal_g1 -> g1 etc
            self.rename_metacal(data)

            #  Now output the calibrated bin data for this processor
            for b in my_bins:

                # Select objects to go in this bin
                if b == "all":
                    # the 2D case is any object from any other bin
                    w = np.where(data["bin"] >= 0)
                    cal = cal2d
                else:
                    # otherwise just objects in this bin
                    w = np.where(data["bin"] == b)
                    cal = cals[b]

                # Cut down the data to just this selection for output
                d = {name: data[name][w] for name in output_cols}

                # Calibrate the shear columns 
                if cat_type=='hsc':
                    d["g1"], d["g2"] = cal.apply(
                        d["g1"], d["g2"], d["c1"], d["c2"]
                    )
                elif cat_type=='lensfit':
                    d["g1"], d["g2"] = cal.apply(
                        d["dec"],d["g1"], d["g2"], subtract_mean=subtract_mean_shear
                    )
                else:
                    d["g1"], d["g2"] = cal.apply(
                    d["g1"], d["g2"]
                )

                # Write output, keeping track of sizes
                splitter.write_bin(d, b)

        splitter.finish(my_bins)
        output_file.close()

    def setup_output(self, extra_cols):
        # count the expected number of objects per bin from the tomo data
        with self.open_input("shear_tomography_catalog") as f:
            counts = f["tomography/counts"][:]
            count2d = f["tomography/counts_2d"][0]
            nbin = len(counts)

        # Prepare the calibrated output catalog
        f = self.open_output("binned_shear_catalog", parallel=True)

        #  we only retain these columns
        cols = ["ra", "dec", "weight", "g1", "g2"] + extra_cols

        # structure is /shear/bin_1, /shear/bin_2, etc
        g = f.create_group("shear")

        # These are both the same here, but there may be some stages
        # that are still expecting it to be called "nbin_source"
        g.attrs["nbin"] = nbin
        g.attrs["nbin_source"] = nbin

        # This maps the bin numbers (and name, in the case
        # of the non-tomographic "all" bin) to the number
        # of objects in each, and is used by the splitter
        # to initialize the output groups.
        bins = {b: c for b, c in enumerate(counts)}
        bins["all"] = count2d
        # These are the possible integer columns
        dtypes = {"id": "i8", "flags": "i8"}
        splitter = Splitter(g, "bin", cols, bins, dtypes=dtypes)

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
