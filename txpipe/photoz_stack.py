from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, TomographyCatalog, HDFFile 
import numpy as np

class TXPhotozStack(PipelineStage):
    """
    Naively stack photo-z PDFs in bins according to previous selections.

    """
    name='TXPhotozStack'
    inputs = [
        ('photoz_pdfs', PhotozPDFFile),
        ('tomography_catalog', TomographyCatalog)
    ]
    outputs = [
        ('photoz_stack', HDFFile), # Haven't worked out a proper format for this yet
            
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

        # Set up the array we will stack the PDFs into
        # first get the sizes from metadata
        z, nbin_source, nbin_lens = self.get_metadata()
        nz = len(z)
        source_pdfs = np.zeros((nbin_source, nz))
        lens_pdfs = np.zeros((nbin_lens, nz))
        source_counts = np.zeros(nbin_source)
        lens_counts = np.zeros(nbin_lens)


        # We use two iterators to loop through the data files.
        # These load the data chunk by chunk instead of all at once
        # iterate_hdf is a method that the superclass defines.

        # We are implicitly assuming here that the files are the same
        # length (that they match up row for row)
        photoz_iterator = self.iterate_hdf(
            'photoz_pdfs', # tag of input file to iterate through
            'pdf', # data group within file to look at
            ['pdf'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )

        tomography_iterator = self.iterate_hdf(
            'tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['source_bin', 'lens_bin'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )


        # So we just do a single loop through the pair of files.
        for (_, _, pz_data), (s, e, tomo_data) in zip(photoz_iterator, tomography_iterator):
            # pz_data and tomo_data are dictionaries with the keys as column names and the 
            # values as numpy arrays with a chunk of data (chunk_rows long) in.
            # Each iteration through the loop we get a new chunk.
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # The method also yields the start and end positions in the file.  We don't need those
            # here because we are just summing them all together.  That's what the underscores
            # are above.


            # Now for each tomographic bin find all the objects in that bin.
            # There is probably a better way of doing this.
            for b in range(nbin_source):
                w = np.where(tomo_data['source_bin']==b)

                # Summ all the PDFs from that bin
                source_pdfs[b] += pz_data['pdf'][w].sum(axis=0)
                source_counts[b] += w[0].size

            for b in range(nbin_lens):
                w = np.where(tomo_data['lens_bin']==b)

                # Summ all the PDFs from that bin
                lens_pdfs[b] += pz_data['pdf'][w].sum(axis=0)
                lens_counts[b] += w[0].size

        if self.comm:
            self.reduce(source_pdfs)
            self.reduce(lens_pdfs)
            self.reduce(source_counts)
            self.reduce(lens_counts)

        if self.rank==0:
            # Normalize the stacks
            for b in range(nbin_source):
                source_pdfs[b] /= source_counts[b]
            for b in range(nbin_lens):
                lens_pdfs[b] /= lens_counts[b]


            # And finally save the outputs
            f = self.open_output("photoz_stack")        
            self.save_result(f, "source", nbin_source, z, source_pdfs, source_counts)
            self.save_result(f, "lens", nbin_lens, z, lens_pdfs, lens_counts)
            f.close()

    def reduce(self, x):
        y = np.zeros_like(x) if self.rank==0 else None
        self.comm.Reduce(x,y)
        x[:]=y


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
        z = photoz_file['pdf/z'][:]
        photoz_file.close()

        # Save again but for the number of bins in the tomography
        # catalog
        tomo_file = self.open_input('tomography_catalog')
        nbin_source = tomo_file['tomography'].attrs['nbin_source']
        nbin_lens = tomo_file['tomography'].attrs['nbin_lens']
        tomo_file.close()

        return z, nbin_source, nbin_lens



    def save_result(self, outfile, name, nbin, z, stacked_pdfs, counts):
        """
        Save the computed stacked photo-z bin n(z)
        
        Parameters
        ----------

        nbin: int
            Number of bins

        z: array of shape (nz,)
            redshift axis 

        stacked_pdfs: array of shape (nbin,nz)
            n(z) per bin

        """
        # This is another parent method.  It will open the result
        # as an HDF file which we then deal with.

        # Create a group inside for the n_of_z data we made here.
        group = outfile.create_group(f"n_of_z/{name}")

        # HDF has "attributes" which are for small metadata like this
        group.attrs["nbin"] = nbin
        group.attrs["nz"] = len(z)

        # Save the redshift sampling
        group.create_dataset("z", data=z)
        
        # And all the bins separately
        for b in range(nbin):
            group.attrs[f"count_{b}"] = counts[b]
            group.create_dataset(f"bin_{b}", data=stacked_pdfs[b])






