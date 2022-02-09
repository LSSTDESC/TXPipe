from .base_stage import PipelineStage
from .data_types import ShearCatalog, HDFFile, TextFile, TomographyCatalog, NOfZFile
from .photoz_stack import Stack
from .utils import rename_iterated
import numpy as np
import os


class TXDirectCalibration(PipelineStage):
    name = "TXDirectCalibration"

    inputs = [
        ("calibration_table", TextFile),
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]

    outputs = [("lens_photoz_stack", NOfZFile)]

    config_options = {
        "n_neighbors": 10,
        "metric": "euclidean",
        "algorithm": "kd_tree",
        "bands": "ugrizy",
        "leafsize": 40,
        "distance_delta": 1e-6,
        "nz": 300,
        "zmax": 3.0,
        "chunk_rows": 100_000,
    }

    def run(self):
        import sklearn.neighbors
        import scipy.spatial

        # Read and process the spectroscopic sample
        spec_data, spec_z, spec_dist, spec_weights = self.read_spectroscopic_sample()

        # make the stack we need. Mostly we actually just use this to keep
        # track of the number of bins and the z range and stuff like that
        stack = self.setup_stack()

        # These are the weights on each spectroscopic galaxy, which we will
        # build up below. We have a different set of weights for each tomographic bin
        weights = np.zeros((stack.nbin, spec_z.size))

        # Loop through the input data, a chunk at a time
        for s, e, photo_data in self.data_iterator():
            print(f"Rank {self.rank} processing rows {s} - {e}")
            # accumulate the weights for this chunk of data
            weights += self.get_weights(stack.nbin, photo_data, spec_data, spec_dist)

        # Sum all the weights across processors
        if self.comm is not None:
            self.comm.Barrier()
            in_place_reduce(weights, self.comm)

        # Save results to our output file
        self.save_results(stack, weights, spec_z, spec_weights)

    def save_results(self, stack, weights, spec_z, spec_weights):
        # Only the root process saves the data
        if self.rank != 0:
            return

        # Make the final n(z) calculation, using a weighted histogram of the
        # spectroscopic objects.
        for i in range(stack.nbin):
            stack.stack[i], _ = np.histogram(
                spec_z,
                bins=stack.nz,
                range=(0, self.config["zmax"]),
                weights=weights[i] * spec_weights,
            )

        # Save the result to our chosen file
        with self.open_output("lens_photoz_stack") as f:
            stack.save(f)

    def setup_stack(self):
        # Get the number of tomographic bins we need
        with self.open_input("lens_tomography_catalog") as f:
            nbin = f["tomography"].attrs["nbin_lens"]

        # Set up the z grid and the stack object which collects
        # together the n(z) for the different bins
        z = np.linspace(0, self.config["zmax"], self.config["nz"])
        stack = Stack("lens", z, nbin)
        return stack

    def data_iterator(self):
        # Load magnitude columns and corresponding
        # lens bin and weight columns
        photo_cols = [f"mag_{b}" for b in self.config["bands"]]
        lens_cols = ["lens_bin", "lens_weight"]

        # Rename the lens_* columns to just *. This is so that
        # we can use a subclass for sources later, unmodified.
        renames = {"lens_bin": "bin", "lens_weight": "weight"}

        # This is a generator function - it returns a new chunk
        # of data each step in the for loop we call it in.
        return rename_iterated(
            self.combined_iterators(
                self.config["chunk_rows"],
                "photometry_catalog",
                "photometry",
                photo_cols,
                "lens_tomography_catalog",
                "tomography",
                lens_cols,
            ),
            renames,
        )

    def read_spectroscopic_sample(self):
        from sklearn.neighbors import NearestNeighbors
        from astropy.table import Table

        bands = self.config["bands"]

        # For testing we just use a sample "spectroscopy" file
        # in text form. Eventually we should replace that with
        # something from the PZ group
        spectro_sample_file = self.get_input("calibration_table")
        data_set = Table.read(spectro_sample_file, format="ascii")

        # pull out the spec-z and weight columns,
        spec_z = np.array(data_set["sz"])

        # There may not be a weight column. Use all 1 if not.
        if "weight" in data_set.colnames:
            print("Found a spectroscopic weight column")
            weights = np.array(data_set["weight"])
        else:
            print("No spectroscopic weights found: using equal weights")
            weights = np.ones_like(spec_z)

        # Get the magnitude data out and put it in the right shape
        # for the nearest neighbors bit
        mags = np.array([data_set[b] for b in bands]).T

        # Find nearest neighbors in color space to the 10th-nearest other
        # spec-z sample. We use this radius as an inverse proxy for the
        # density of the spec-z points locally.
        # The 10 is configurable, and for test data where there are not
        # many photometric data points you will probably have to increase it.
        if self.rank == 0:
            print("Preparing spectroscopic data")

        neighbors = NearestNeighbors(
            n_neighbors=self.config["n_neighbors"],
            algorithm=self.config["algorithm"],
            metric=self.config["metric"],
        ).fit(mags)

        distances, _ = neighbors.kneighbors(mags)
        distances = np.amax(distances, axis=1) + self.config["distance_delta"]

        if self.rank == 0:
            print("    ... done.")

        return mags, spec_z, distances, weights

    def get_weights(self, nbin, photo_data, spec_data, spec_dist):
        import scipy.spatial

        bands = self.config["bands"]
        weight = photo_data["weight"]

        spec_weights = np.zeros((nbin, spec_dist.size))

        for i in range(nbin):
            # Get the chunk of the photometric data for this tomographic bin
            sel = photo_data["bin"] == i
            d = np.array([photo_data[f"mag_{b}"][sel] for b in bands]).T

            # TODO: deal with inf (too faint) and nan (unmeasured) properly.
            # This is mentioned as an issue in Hildebrandt et al 2017
            d[~np.isfinite(d)] = 40

            # Make the tree for the photometric data, and, for each spec-z sample,
            # find all the photo-z galaxies nearby that sample. Where "nearby" is
            # defined as the distance to the 10th nearest other spec-z sample
            # (we calculated this above)
            tree = scipy.spatial.KDTree(d, leafsize=self.config["leafsize"])
            indices = tree.query_ball_point(spec_data, spec_dist)

            # indices is an array of lists, so we can't do anything more numpy-ish
            # than this, as far as I can see.
            for j, index in enumerate(indices):
                spec_weights[i, j] += weight[index].sum()

        return spec_weights
