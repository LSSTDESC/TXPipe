from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, ShearCatalog, YamlFile, HDFFile, DataFile
import sys
import numpy as np


class PZPDFMLZ(PipelineStage):
    """
    Generate photo-z PDFs using MLZ

    This is deprecated; use the RAIL stages.
    """

    name = "PZPDFMLZ"
    parallel = False
    inputs = [
        ("photometry_catalog", HDFFile),
        ("photoz_trained_model", DataFile),
    ]
    outputs = [
        ("lens_photoz_pdfs", PhotozPDFFile),
    ]

    config_options = {"zmax": float, "nz": int, "chunk_rows": 10000, "bands": "ugrizy"}

    def run(self):
        """ """
        import mlz_desc
        import mlz_desc.ml_codes
        import scipy.stats

        zmax = self.config["zmax"]
        nz = self.config["nz"]
        z = np.linspace(0.0, zmax, nz)

        # Open the input catalog and check how many objects
        # we will be running on.
        cat = self.open_input("photometry_catalog")
        nobj = cat["photometry/ra"].size
        cat.close()

        features, trees = self.load_training()

        # Prepare the output HDF5 file
        output_file = self.prepare_output(nobj, z)

        bands = self.config["bands"]
        # The columns we need to calculate the photo-z.
        # Note that we need all the metacalibrated variants too.
        cols = [f"mag_{band}" for band in bands]

        # Loop through chunks of the data.
        # Parallelism is handled in the iterate_input function -
        # each processor will only be given the sub-set of data it is
        # responsible for.  The HDF5 parallel output mode means they can
        # all write to the file at once too.
        chunk_rows = self.config["chunk_rows"]
        for start, end, data in self.iterate_hdf(
            "photometry_catalog", "photometry", cols, chunk_rows
        ):
            print(f"Process {self.rank} running photo-z for rows {start}-{end}")
            sys.stdout.flush()
            # Compute some mock photo-z PDFs and point estimates
            pdfs, point_estimates = self.calculate_photozs(data, z, features, trees)
            # Save this chunk of data to the output file
            self.write_output(output_file, start, end, pdfs, point_estimates)

        # Synchronize processors
        if self.is_mpi():
            self.comm.Barrier()

        # Finish
        output_file.close()

    def load_training(self):
        import mlz_desc
        import mlz_desc.ml_codes
        import sys

        sys.modules["mlz"] = sys.modules["mlz_desc"]
        filename = self.get_input("photoz_trained_model")
        features, trees = np.load(filename, allow_pickle=True)
        return features, trees

    def calculate_photozs(self, data, z, features, trees):
        """
        Generate random photo-zs.

        This is a mock method that instead of actually
        running any photo-z analysis just spits out some random PDFs.

        This method is run on chunks of data, not the whole thing at
        once.

        It does however generate outputs in the right format to be
        saved later, and generates point estimates, used for binning
        and assumed to be a mean or similar statistic from each bin,
        for each of the five metacalibrated variants of the magnitudes.

        Parameters
        ----------

        data: dict of arrays
            Chunk of input photometry catalog containing object magnitudes

        z: array
            The redshift values at which to "compute" P(z) values

        Returns
        -------

        pdfs: array of shape (n_chunk, n_z)
            The output PDF values

        point_estimates: array of shape (5, n_chunk)
            Point-estimated photo-zs for each of the 5 metacalibrated variants

        """
        import numpy as np
        import scipy.stats

        # Number of z points we will be using
        nbin = len(z) - 1
        nrow = len(data["mag_i"])

        # These are the old names for the features
        if features == [
            "mag_u_lsst",
            "mag_g_lsst",
            "mag_r_lsst",
            "mag_i_lsst",
            "mag_z_lsst",
            "mag_y_lsst",
            "mag_u_lsst-mag_g_lsst",
            "mag_g_lsst-mag_r_lsst",
            "mag_r_lsst-mag_i_lsst",
            "mag_i_lsst-mag_z_lsst",
            "mag_z_lsst-mag_y_lsst",
        ]:
            x = [data[f"mag_{b}"] for b in "ugrizy"]

            ug = data["mag_u"] - data["mag_g"]
            gr = data["mag_g"] - data["mag_r"]
            ri = data["mag_r"] - data["mag_i"]
            iz = data["mag_i"] - data["mag_z"]
            zy = data["mag_z"] - data["mag_y"]
            x += [ug, gr, ri, iz, zy]

        elif features == [
            "mag_u_lsst",
            "mag_g_lsst",
            "mag_r_lsst",
            "mag_i_lsst",
            "mag_u_lsst-mag_g_lsst",
            "mag_g_lsst-mag_r_lsst",
            "mag_r_lsst-mag_i_lsst",
            "mag_i_lsst-mag_z_lsst",
            "mag_z_lsst-mag_y_lsst",
        ]:
            x = [data[f"mag_{b}"] for b in "ugriz"]
            ug = data["mag_u"] - data["mag_g"]
            gr = data["mag_g"] - data["mag_r"]
            ri = data["mag_r"] - data["mag_i"]
            iz = data["mag_i"] - data["mag_z"]
            zy = data["mag_z"] - data["mag_y"]
            x += [ug, gr, ri, iz, zy]
        else:
            raise ValueError("Need to re-code for the features you used")

        x = np.vstack(x).T

        pdfs = np.empty((nrow, nbin))
        point_estimates = np.empty(nrow)

        for i in range(nrow):
            # Run all the tree regressors on each of the metacal
            # variants
            values = np.concatenate([T.get_vals(x[i]) for T in trees]).ravel()
            pdfs[i], _ = np.histogram(values, bins=z)
            pdfs[i] /= pdfs[i].sum()
            point_estimates[i] = np.mean(values)
        return pdfs, point_estimates

    def write_output(self, output_file, start, end, pdfs, point_estimates):
        """
        Write out a chunk of the computed PZ data.

        Parameters
        ----------

        output_file: h5py.File
            The object we are writing out to

        start: int
            The index into the full range of data that this chunk starts at

        end: int
            The index into the full range of data that this chunk ends at

        pdfs: array of shape (n_chunk, n_z)
            The output PDF values

        point_estimates: array of shape (5, n_chunk)
            Point-estimated photo-zs for each of the 5 metacalibrated variants

        """
        group1 = output_file["pdf"]
        group1["pdf"][start:end] = pdfs
        group2 = output_file["point_estimates"]
        group2["z_mean"][start:end] = point_estimates

    def prepare_output(self, nobj, z):
        """
        Prepare the output HDF5 file for writing.

        Note that this is done by all the processes if running in parallel;
        that is part of the design of HDF5.

        Parameters
        ----------

        nobj: int
            Number of objects in the catalog

        z: array
            Points on the redshift axis that the PDF will be evaluated at.

        Returns
        -------
        f: h5py.File object
            The output file, opened for writing.

        """
        # Open the output file.
        # This will automatically open using the HDF5 mpi-io driver
        # if we are running under MPI and the output type is parallel
        f = self.open_output("lens_photoz_pdfs", parallel=True)

        z_mid = 0.5 * (z[1:] + z[:-1])
        # Create the space for output data
        nz = len(z_mid)
        group1 = f.create_group("pdf")
        group1.create_dataset("zgrid", (nz,), dtype="f4")
        group1.create_dataset("pdf", (nobj, nz), dtype="f4")
        group2 = f.create_group("point_estimates")
        group2.create_dataset("z_mean", (nobj,), dtype="f4")

        # One processor writes the redshift axis to output.
        if self.rank == 0:
            group1["zgrid"][:] = z_mid

        return f


if __name__ == "__main__":
    PipelineStage.main()
