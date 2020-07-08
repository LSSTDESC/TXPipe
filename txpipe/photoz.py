from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, ShearCatalog, YamlFile, HDFFile
import numpy as np


class TXRandomPhotozPDF(PipelineStage):
    """
    This is a placeholder for an actual photoz pipeline!

    At the moment it just randomly generates a log-normal PDF for each object.

    The pipeline loops through input photometry data,
    "calculating" (at random!) a PDF and a point-estimate for each row.
    It must generate the point estimates for the five different 
    metacal variants (which each have different shears applied). 

    It can do this in parallel if needed.

    We might want to move some of the functionality here (e.g. the I/O)
    into a general parent class.

    """

    name = 'TXRandomPhotozPDF'

    inputs = [
        ('photometry_catalog', HDFFile),
    ]
    outputs = [
        ('photoz_pdfs', PhotozPDFFile),
    ]

    # Configuration options.
    # If the value here is a type, like "float" or "int" then that
    # means there is no default value for that parameter and the
    # user must include the parameter in the config file, of that type.
    # Otherwise the entry lists the default value for the parameter.
    config_options = {'zmax': float, 'nz': int, 'chunk_rows': 10000, 'bands': 'ugriz'}

    def run(self):
        """
        Run the analysis for this stage.

         - prepares the output HDF5 file
         - loads in chunks of input data, one at a time
         - computes mock photo-z PDFs for each chunk
         - writes each chunk to output
         - closes the output file

        """
        import scipy.stats

        zmax = self.config['zmax']
        nz = self.config['nz']
        z = np.linspace(0.0, zmax, nz)

        # Open the input catalog and check how many objects
        # we will be running on.
        cat = self.open_input("photometry_catalog")
        nobj = cat['photometry/id'].size
        cat.close()

        # Prepare the output HDF5 file
        output_file = self.prepare_output(nobj, z)

        suffices = ["", "_1p", "_1m", "_2p", "_2m"]
        bands = self.config['bands']
        # The columns we need to calculate the photo-z.
        # Note that we need all the metacalibrated variants too.
        cols = [f'mag_{band}_lsst{suffix}' for band in bands for suffix in suffices]

        # Loop through chunks of the data.
        # Parallelism is handled in the iterate_input function -
        # each processor will only be given the sub-set of data it is
        # responsible for.  The HDF5 parallel output mode means they can
        # all write to the file at once too.
        chunk_rows = self.config['chunk_rows']
        for start, end, data in self.iterate_hdf(
            'photometry_catalog', "photometry", cols, chunk_rows
        ):
            print(f"Process {self.rank} running photo-z for rows {start}-{end}")

            # Compute some mock photo-z PDFs and point estimates
            pdfs, point_estimates = self.calculate_photozs(data, z)
            # Save this chunk of data to the output file
            self.write_output(output_file, start, end, pdfs, point_estimates)

        # Synchronize processors
        if self.is_mpi():
            self.comm.Barrier()

        # Finish
        output_file.close()

    def calculate_photozs(self, data, z):
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
        import scipy.stats

        # Number of z points we will be using
        nz = self.config['nz']

        # We just want any random element to use as the size.
        # It's painful how ugly this is in python 3!
        nobj = len(next(iter(data.values())))

        # Generate some random median redshifts between 0.2 and 1.0
        medians = np.random.uniform(0.2, 1.0, size=nobj)
        sigmas = 0.05 * (1 + medians)

        # Make the array which will contain this chunk of PDFs
        pdfs = np.empty((nobj, nz), dtype='f4')

        # Note that we need metacalibrated versions of
        # the point estimates.  That's why the 5 is there.
        point_estimates = np.empty((5, nobj), dtype='f4')

        # Loop through each object and make a fake PDF
        # for it, saving it to the output space
        for i, (mu, sigma) in enumerate(zip(medians, sigmas)):
            pdf = scipy.stats.lognorm.pdf(z, s=sigma, scale=mu)
            pdfs[i] = pdf
            point_estimates[:, i] = mu

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
        group = output_file['pdf']
        group['pdf'][start:end] = pdfs
        group['mu'][start:end] = point_estimates[0]
        group['mu_1p'][start:end] = point_estimates[1]
        group['mu_1m'][start:end] = point_estimates[2]
        group['mu_2p'][start:end] = point_estimates[3]
        group['mu_2m'][start:end] = point_estimates[4]

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
        f = self.open_output('photoz_pdfs', parallel=True)

        # Create the space for output data
        nz = len(z)
        group = f.create_group('pdf')
        group.create_dataset("z", (nz,), dtype='f4')
        group.create_dataset("pdf", (nobj, nz), dtype='f4')
        group.create_dataset("mu", (nobj,), dtype='f4')
        group.create_dataset("mu_1p", (nobj,), dtype='f4')
        group.create_dataset("mu_1m", (nobj,), dtype='f4')
        group.create_dataset("mu_2p", (nobj,), dtype='f4')
        group.create_dataset("mu_2m", (nobj,), dtype='f4')

        # One processor writes the redshift axis to output.
        if self.rank == 0:
            group['z'][:] = z

        return f


if __name__ == '__main__':
    PipelineStage.main()
