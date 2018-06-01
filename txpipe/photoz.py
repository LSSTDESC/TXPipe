from ceci import PipelineStage
from descformats.tx import PhotozPDFFile, MetacalCatalog, YamlFile, HDFFile

class TXRandomPhotozPDF(PipelineStage):
    """
    This is a placeholder for an actual photoz pipeline!

    At the moment it just randomly generates a log-normal PDF for each object.
    Hopefully the real pipeline will be more accurate than that.

    """
    name='TXRandomPhotozPDF'
    inputs = [
        ('photometry_catalog', HDFFile),
    ]
    outputs = [
        ('photoz_pdfs', PhotozPDFFile),
    ]

    # Configuration options.  If the value is not a type then it specifies a default value
    config_options = {'zmax': float, 'nz': int, 'chunk_rows': 10000, 'bands':'ugriz'}


    def run(self):
        """
        The run method is where all the work of the stage is done.
        In this case it:
         - reads the config file
         - prepares the output HDF5 file
         - loads in chunks of input data, one at a time
         - computes mock photo-z PDFs for each chunk
         - writes each chunk to output
         - closes the output file

        """
        import numpy as np
        import fitsio

        config = self.config
        z = np.linspace(0.0, config['zmax'], config['nz'])
        
        # Open the input catalog and check how many objects
        # we will be running on.
        cat = self.open_input("photometry_catalog")
        nobj = cat['photometry/id'].size
        cat.close()
        
        # Prepare the output HDF5 file
        output_file = self.prepare_output(nobj, config, z)

        suffices = ["", "_1p", "_1m", "_2p", "_2m"]
        bands = config['bands']
        # The columns we need to calculate the photo-z.
        # Note that we need all the metacalibrated variants too.
        cols = [f'mag_{band}_lsst{suffix}' for band in bands for suffix in suffices]

        # Loop through chunks of the data.
        # Parallelism is handled in the iterate_input function - 
        # each processor will only be given the sub-set of data it is 
        # responsible for.  The HDF5 parallel output mode means they can
        # all write to the file at once too.
        chunk_rows = config['chunk_rows']
        for start, end, data in self.iterate_hdf('photometry_catalog', "photometry", cols, chunk_rows):
            print(f"Process {self.rank} running photo-z for rows {start}-{end}")

            # Compute some mock photo-z PDFs and point estimates
            pdfs, point_estimates = self.calculate_photozs(data, z, config)
            # Save this chunk of data to the output file
            self.write_output(output_file, start, end, pdfs, point_estimates)

        # Synchronize processors
        if self.is_mpi():
            self.comm.Barrier()

        # Finish
        output_file.close()

    def calculate_photozs(self, data, z, config):
        # Mock photo-z code generating random PDFs.
        # Note that we need metacalibrated versions of
        # the point estimates.  That's why the 5 is there.
        import numpy as np
        import scipy.stats
        nz = config['nz']
        # painful how ugly this is in python 3 - we just want any random element
        nobj = len(next(iter(data.values())))
        medians = np.random.uniform(0.2, 1.0, size=nobj)
        sigmas = 0.05 * (1+medians)
        pdfs = np.empty((nobj,nz), dtype='f4')
        point_estimates = np.empty((5,nobj), dtype='f4')
        for i,(mu,sigma) in enumerate(zip(medians,sigmas)):
            pdf = scipy.stats.lognorm.pdf(z, s=sigma, scale=mu)
            pdfs[i] = pdf
            point_estimates[:,i] = mu
        return pdfs, point_estimates

    def write_output(self, output_file, start, end, pdfs, point_estimates):
        group = output_file['pdf']
        group['pdf'][start:end] = pdfs
        group['mu'][start:end] = point_estimates[0]
        group['mu_1p'][start:end] = point_estimates[1]
        group['mu_1m'][start:end] = point_estimates[2]
        group['mu_2p'][start:end] = point_estimates[3]
        group['mu_2m'][start:end] = point_estimates[4]





    def prepare_output(self, nobj, config, z):
        # Open the output file.
        # This will automatically open using the HDF5 mpi-io driver 
        # if we are running under MPI and the output type is parallel
        f = self.open_output('photoz_pdfs', parallel=True)
            

        nz = config['nz']
        group = f.create_group('pdf')
        group.create_dataset("z", (nz,), dtype='f4')
        group.create_dataset("pdf", (nobj,nz), dtype='f4')
        group.create_dataset("mu", (nobj,), dtype='f4')
        group.create_dataset("mu_1p", (nobj,), dtype='f4')
        group.create_dataset("mu_1m", (nobj,), dtype='f4')
        group.create_dataset("mu_2p", (nobj,), dtype='f4')
        group.create_dataset("mu_2m", (nobj,), dtype='f4')
        group['z'][:] = z
        return f


if __name__ == '__main__':
    PipelineStage.main()
