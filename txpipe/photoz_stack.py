from ceci import PipelineStage
from descformats.tx import PhotozPDFFile, TomographyCatalog, HDFFile 

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
        import numpy as np


        # Set up the array we will stack the PDFs into
        # first get the sizes from metadata
        z, nbin = self.get_metadata()
        nz = len(z)
        stacked_pdfs = np.zeros((nbin, nz))
        counts = np.zeros(nbin)


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
            ['bin'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )

        # At the moment I'm not trying to do this in parallel - it doesn't seem worth it.
        # If we wanted to do that we'd have to gather the stacked pdfs together at the end
        # and sum them.
        if self.comm is not None:
            raise ValueError("The code TXPhotozStack is not designed to be run in parallel")

        # So we just do a single loop through the pair of files.
        for (_, _, pz_data), (_, _, tomo_data) in zip(photoz_iterator, tomography_iterator):
            # pz_data and tomo_data are dictionaries with the keys as column names and the 
            # values as numpy arrays with a chunk of data (chunk_rows long) in.
            # Each iteration through the loop we get a new chunk.

            # The method also yields the start and end positions in the file.  We don't need those
            # here because we are just summing them all together.  That's what the underscores
            # are above.


            # Now for each tomographic bin find all the objects in that bin.
            # There is probably a better way of doing this.
            for b in range(nbin):
                w = np.where(tomo_data['bin']==b)

                # Summ all the PDFs from that bin
                stacked_pdfs[b] += pz_data['pdf'][w].sum(axis=0)
                counts[b] += w[0].size


        # Normalize the stacks
        for b in range(nbin):
            stacked_pdfs[b] /= counts[b]


        # And finally save the outputs
        self.save_result(nbin, z, stacked_pdfs)

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
        nbin = tomo_file['tomography'].attrs['nbin']
        tomo_file.close()

        return z, nbin



    def save_result(self, nbin, z, stacked_pdfs):
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
        f = self.open_output("photoz_stack")

        # Create a group inside for the n_of_z data we made here.
        group = f.create_group("n_of_z")

        # HDF has "attributes" which are for small metadata like this
        group.attrs["nbin"] = nbin
        group.attrs["nz"] = len(z)

        # Save the redshift sampling
        group.create_dataset("z", data=z)
        
        # And all the bins separately
        for b in range(nbin):
            group.create_dataset(f"bin_{b}", data=stacked_pdfs[b])

        # And tidy up.
        f.close()





