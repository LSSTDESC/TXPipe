from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, TomographyCatalog, HDFFile, PNGFile, NOfZFile
from .utils.mpi_utils import in_place_reduce
import numpy as np
import warnings

class Stack:
    def __init__(self, name, z, nbin):
        """
        Create an n(z) stacker

        Parameters
        ----------
        name: str
            Name of the stack, for when we save it
        z: array
            redshift edges array
        nbin: int
            number of tomographic bins
        """
        self.name = name
        self.z = z
        self.nbin = nbin
        self.nz = z.size
        self.stack = np.zeros((nbin, z.size))
        self.counts = np.zeros(nbin)

    def add_pdfs(self, bins, pdfs):
        """
        Add a set of PDFs to the stack, per tomographic bin

        Parameters
        ----------
        bins: array[int]
            The tomographic bin for each object
        pdfs: 2d array[float]
            The p(z) per object
        """
        for b in range(self.nbin):
            w = np.where(bins==b)
            self.stack[b] += pdfs[w].sum(axis=0)
            self.counts[b] += w[0].size

    def add_delta_function(self, bins, z):
        """
        Add a set of objects to the stack whose redshift
        is known perfectly

        Parameters
        ----------
        bins: array[int]
            The tomographic bin for each object
        z: array[float]
            The redshift per object
        """
        # here self.z are the edges of the bins the nz will have. 
        stack_bin = np.digitize(z, self.z)
        for b in range(self.nbin):
            w = np.where(bins==b)
            stack_bin_b = stack_bin[w]
            for i in stack_bin_b:
                if 0 <= i < self.nz:
                    # digitize returns numbers between 1 and len(self.z)-1
                    # that is why we need to subtract 1 here, since the
                    # minimum value of i will be 1. 
                    self.stack[b][i-1] += 1
                    self.counts[b] += 1

    def save(self, outfile, comm=None):
        """
        Write this stack to a new group in the output file.
        Collect the stack from all processors if comm is provided

        Parameters
        ----------
        outfile: h5py.File
            Output file, already open

        comm: mpi4py communicator
            Optional, default=0
        """
        # stack the results from different comms
        if comm is not None:
            in_place_reduce(self.stack, comm)
            in_place_reduce(self.counts, comm)

            # only root saves output
            if comm.Get_rank() != 0:
                return

        # normalize each n(z)
        for b in range(self.nbin):
            if self.counts[b] > 0:
                self.stack[b] /= self.counts[b]

        # Create a group inside for the n_of_z data we made here.
        group = outfile.create_group(f"n_of_z/{self.name}")

        # HDF has "attributes" which are for small metadata like this
        group.attrs["nbin"] = self.nbin
        group.attrs["nz"] = len(self.z)

        # Save the redshift sampling. Adding half a bin here to save the mean z.
        # remove the last item when converting from edges to mean.
        group.create_dataset("z", data=self.z[:-1]+(self.z[2]-self.z[1])/2.)
        
        # And all the bins separately
        for b in range(self.nbin):
            group.attrs[f"count_{b}"] = self.counts[b]
            # remove the last item
            group.create_dataset(f"bin_{b}", data=self.stack[b][:-1])



class TXPhotozSourceStack(PipelineStage):
    """
    Naively stack photo-z PDFs in bins according to previous selections.

    This parent class does only the source bins.
    """
    name = "TXPhotozSourceStack"

    inputs = [
        ('source_photoz_pdfs', PhotozPDFFile),
        ('shear_tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('shear_photoz_stack', NOfZFile),            
    ]
    config_options = {
        'chunk_rows': 5000,  # number of rows to read at once
    }

    def run(self):
        """
        Run the analysis for this stage.
        
         - Get metadata and allocate space for output
         - Set up iterators to loop through tomography and PDF input files
         - Accumulate the PDFs for each object in each bin
         - Divide by the counts to get the stacked PDF
        """

        # Create the stack objects
        outputs = self.prepare_outputs('source')
        warnings.warn("WEIGHTS/RESPONSE ARE NOT CURRENTLY INCLUDED CORRECTLY in PZ STACKING")

        # So we just do a single loop through the pair of files.
        for (s, e, data) in self.data_iterator():
            # Feed back on our progress
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Add data to the stacks
            self.stack_data('source', data, outputs)
        # Save the stacks
        self.write_outputs("shear_photoz_stack", outputs)

    def prepare_outputs(self, name):
        z, nbin_source = self.get_metadata()
        # For this class we do two stacks, and main one and a 2d one
        stack = Stack(name, z, nbin_source)
        stack2d = Stack(f'{name}2d', z, 1)
        return stack, stack2d


    def data_iterator(self):
        # This collects together matching inputs from the different
        # input files and returns an iterator to them which yields
        # start, end, data
        return self.combined_iterators(
                self.config['chunk_rows'],
                'source_photoz_pdfs', # tag of input file to iterate through
                'pdf', # data group within file to look at
                ['pdf'], # column(s) to read

                'shear_tomography_catalog', # tag of input file to iterate through
                'tomography', # data group within file to look at
                ['source_bin'], # column(s) to read
        )

    def stack_data(self, name, data, outputs):
        # add the data we have loaded into the stacks
        stack, stack2d = outputs
        stack.add_pdfs(data[f'{name}_bin'], data['pdf'])
        # -1 indicates no selection.  For the non-tomo 2d case
        # we just say anything that is >=0 is set to bin zero, like this
        bin2d = data[f'{name}_bin'].clip(-1, 0)
        stack2d.add_pdfs(bin2d, data['pdf'])


    def write_outputs(self, tag, outputs):
        source_stack, source2d_stack = outputs
        # only the root process opens the file.
        # The others don't use that so we have to
        # give them something in place
        # (i.e. inside the save method the non-root procs
        # will not reference the first arg)
        if self.rank == 0:
            f = self.open_output(tag)
            stack.save(f, self.comm)
            stack2d.save(f, self.comm)
            f.close()
        else:
            stack.save(None, self.comm)
            stack2d.save(None, self.comm)


    def get_metadata(self):
        """
        Load the z column and the number of bins

        Returns
        -------
        z: array
            Redshift column for photo-z PDFs
        nbin:
            Number of different redshift bins
            to split into.

        """
        # It's a bit odd but we will just get this from the file and 
        # then close it again, because we're going to use the 
        # built-in iterator method to get the rest of the data

        # open_input is a method defined on the superclass.
        # it knows about different file formats (pdf, fits, etc)
        photoz_file = self.open_input('photoz_pdfs')

        # This is the syntax for reading a complete HDF column
        z = photoz_file['pdf/zgrid'][:]
        photoz_file.close()

        # Save again but for the number of bins in the tomography catalog
        shear_tomo_file = self.open_input('shear_tomography_catalog')
        nbin_source = shear_tomo_file['tomography'].attrs['nbin_source']
        shear_tomo_file.close()

        return z, nbin_source



class TXPhotozLensStack(TXPhotozSourceStack):
    """
    Naively stack photo-z PDFs in bins according to previous selections.

    This parent class does only the source bins.
    """
    name = "TXPhotozLensStack"
    inputs = [
        ('lens_photoz_pdfs', PhotozPDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('lens_photoz_stack', NOfZFile),            
    ]
    config_options = {
        'chunk_rows': 5000,  # number of rows to read at once
    }

    def run(self):
        """
        Run the analysis for this stage.
        
         - Get metadata and allocate space for output
         - Set up iterators to loop through tomography and PDF input files
         - Accumulate the PDFs for each object in each bin
         - Divide by the counts to get the stacked PDF
        """

        # Create the stack objects
        outputs = self.prepare_outputs('lens')
        warnings.warn("WEIGHTS/RESPONSE ARE NOT CURRENTLY INCLUDED CORRECTLY in PZ STACKING")

        # So we just do a single loop through the pair of files.
        for (s, e, data) in self.data_iterator():
            # Feed back on our progress
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Add data to the stacks
            self.stack_data('lens', data, outputs)
        # Save the stacks
        self.write_outputs("lens_photoz_stack", outputs)



    def data_iterator(self):
        # This collects together matching inputs from the different
        # input files and returns an iterator to them which yields
        # start, end, data
        return self.combined_iterators(
                self.config['chunk_rows'],
                'lens_photoz_pdfs', # tag of input file to iterate through
                'pdf', # data group within file to look at
                ['pdf'], # column(s) to read

                'lens_tomography_catalog', # tag of input file to iterate through
                'tomography', # data group within file to look at
                ['lens_bin'], # column(s) to read
        )


class TXPhotozPlots(PipelineStage):
    """
    Make n(z) plots

    """
    name = "TXPhotozPlots"
    inputs = [
        ('shear_photoz_stack', NOfZFile),
        ('lens_photoz_stack', NOfZFile)
    ]
    outputs = [
        ('nz_lens', PNGFile),
        ('nz_source', PNGFile),
    ]

    config_options = {
    }


    def run(self):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        f = self.open_input('lens_photoz_stack', wrapper=True)
        
        out1 = self.open_output('nz_lens', wrapper=True)
        f.plot('lens')
        plt.legend(frameon=False)
        plt.title("Lens n(z)")
        plt.xlim(xmin=0)
        out1.close()

        f = self.open_input('shear_photoz_stack', wrapper=True)
        out2 = self.open_output('nz_source', wrapper=True)
        f.plot('source')
        plt.legend(frameon=False)
        plt.title("Source n(z)")
        plt.xlim(xmin=0)
        out2.close()


# class TXTrueNumberDensity(TXPhotozSourceStack):
#     """
#     Fake an n(z) by histogramming the true redshift values for each object.
#     Uses the same method as its parent but loads the data
#     differently and uses the truth redshift as a delta function PDF
#     """
#     name = "TXTrueNumberDensity"
#     inputs = [
#         ('photometry_catalog', HDFFile),
#         ('shear_tomography_catalog', TomographyCatalog),
#         ('lens_tomography_catalog', TomographyCatalog)
#     ]
#     outputs = [
#         ('shear_photoz_stack', NOfZFile),
#         ('lens_photoz_stack', NOfZFile)
#     ]
#     config_options = {
#         'chunk_rows': 5000,  # number of rows to read at once
#         'zmax': float,
#         'nz': int,
#     }


#     def data_iterator(self):
#         return self.combined_iterators(
#                 self.config['chunk_rows'],
#                 'photometry_catalog', # tag of input file to iterate through
#                 'photometry', # data group within file to look at
#                 ['redshift_true'], # column(s) to read

#                 'shear_tomography_catalog', # tag of input file to iterate through
#                 'tomography', # data group within file to look at
#                 ['source_bin'], # column(s) to read

#                 'lens_tomography_catalog', # tag of input file to iterate through
#                 'tomography', # data group within file to look at
#                 ['lens_bin'], # column(s) to read
#             )

#     def stack_data(self, data, outputs):
#         source_stack, source2d_stack, lens_stack = outputs
#         source_stack.add_delta_function(data['source_bin'], data['redshift_true'])
#         bin2d = data['source_bin'].clip(-1, 0)
#         source2d_stack.add_delta_function(bin2d, data['redshift_true'])
#         lens_stack.add_delta_function(data['lens_bin'], data['redshift_true'])


#     def get_metadata(self):
#         # Check we are running on a photo file with redshift_true
#         photo_file = self.open_input('photometry_catalog')
#         has_z = 'redshift_true' in photo_file['photometry'].keys()
#         photo_file.close()
#         if not has_z:
#             msg = ("The photometry_catalog file you supplied does not have a redshift_true column. "
#                    "If you're running on sims you need to make sure to ingest that column from GCR. "
#                    "If you're running on real data then sadly this isn't going to work. "
#                    "Use a different stacking stage."
#                 )
#             raise ValueError(msg)

#         zmax = self.config['zmax']
#         nz = self.config['nz']
#         z = np.linspace(0, zmax, nz)

#         shear_tomo_file = self.open_input('shear_tomography_catalog')
#         nbin_source = shear_tomo_file['tomography'].attrs['nbin_source']
#         shear_tomo_file.close()

#         lens_tomo_file = self.open_input('lens_tomography_catalog')
#         nbin_lens = lens_tomo_file['tomography'].attrs['nbin_lens']
#         lens_tomo_file.close()

#         return z, nbin_source, nbin_lens


class TXSourceTrueNumberDensity(TXPhotozSourceStack):
    """
    Fake an n(z) by histogramming the true redshift values for each object.
    Uses the same method as its parent but loads the data
    differently and uses the truth redshift as a delta function PDF
    """
    name = "TXSourceTrueNumberDensity"
    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('shear_photoz_stack', NOfZFile),
    ]
    config_options = {
        'chunk_rows': 5000,  # number of rows to read at once
        'zmax': float,
        'nz': int,
    }


    def data_iterator(self):
        return self.combined_iterators(
                self.config['chunk_rows'],
                'photometry_catalog', # tag of input file to iterate through
                'photometry', # data group within file to look at
                ['redshift_true'], # column(s) to read

                'shear_tomography_catalog', # tag of input file to iterate through
                'tomography', # data group within file to look at
                ['source_bin'], # column(s) to read
            )

    def stack_data(self, data, outputs):
        source_stack, source2d_stack = outputs
        source_stack.add_delta_function(data['source_bin'], data['redshift_true'])
        bin2d = data['source_bin'].clip(-1, 0)
        source2d_stack.add_delta_function(bin2d, data['redshift_true'])


    def get_metadata(self):
        # Check we are running on a photo file with redshift_true
        photo_file = self.open_input('photometry_catalog')
        has_z = 'redshift_true' in photo_file['photometry'].keys()
        photo_file.close()
        if not has_z:
            msg = ("The photometry_catalog file you supplied does not have a redshift_true column. "
                   "If you're running on sims you need to make sure to ingest that column from GCR. "
                   "If you're running on real data then sadly this isn't going to work. "
                   "Use a different stacking stage."
                )
            raise ValueError(msg)

        zmax = self.config['zmax']
        nz = self.config['nz']
        z = np.linspace(0, zmax, nz)

        shear_tomo_file = self.open_input('shear_tomography_catalog')
        nbin_source = shear_tomo_file['tomography'].attrs['nbin_source']
        shear_tomo_file.close()

        return z, nbin_source
