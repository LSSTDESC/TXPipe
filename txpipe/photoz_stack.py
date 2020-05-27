from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, TomographyCatalog, HDFFile, PNGFile, NOfZFile
import numpy as np
import warnings

class TXPhotozStack(PipelineStage):
    """
    Naively stack photo-z PDFs in bins according to previous selections.

    """
    name='TXPhotozStack'
    inputs = [
        ('photoz_pdfs', PhotozPDFFile),
        ('shear_tomography_catalog', TomographyCatalog),
        ('lens_tomography_catalog', TomographyCatalog)
    ]
    outputs = [
        ('shear_photoz_stack', NOfZFile),
        ('lens_photoz_stack', NOfZFile)
            
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
        # We also set up accumulators for the combined
        # total tomographic bin (2d)
        z, nbin_source, nbin_lens = self.get_metadata()
        nz = len(z)
        source_pdfs = np.zeros((nbin_source, nz))
        source_pdfs_2d = np.zeros(nz)
        lens_pdfs = np.zeros((nbin_lens, nz))
        source_counts = np.zeros(nbin_source)
        source_counts_2d = 0
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

        shear_tomography_iterator = self.iterate_hdf(
            'shear_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['source_bin'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )

        lens_tomography_iterator = self.iterate_hdf(
            'lens_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['lens_bin'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )

        warnings.warn("WEIGHTS/RESPONSE ARE NOT CURRENTLY INCLUDED CORRECTLY in PZ STACKING")


        # So we just do a single loop through the pair of files.
        for (_, _, pz_data), (s1, e1, shear_tomo_data), (s2, e2, lens_tomo_data) in zip(photoz_iterator, shear_tomography_iterator, lens_tomography_iterator):
            # pz_data and tomo_data are dictionaries with the keys as column names and the 
            # values as numpy arrays with a chunk of data (chunk_rows long) in.
            # Each iteration through the loop we get a new chunk.
            print(f"Process {self.rank} read data chunk {s1:,} - {e1:,}")
            # The method also yields the start and end positions in the file.  We don't need those
            # here because we are just summing them all together.  That's what the underscores
            # are above.


            # Now for each tomographic bin find all the objects in that bin.
            # There is probably a better way of doing this.
            for b in range(nbin_source):
                w = np.where(shear_tomo_data['source_bin']==b)

                # Summ all the PDFs from that bin
                source_pdfs[b] += pz_data['pdf'][w].sum(axis=0)
                source_counts[b] += w[0].size

            # For the 2D source bin we take every object that is selected
            # for any tomographic bin (the non-selected objects
            # have bin=-1)s
            w = np.where(shear_tomo_data['source_bin']>=0)
            source_pdfs_2d += pz_data['pdf'][w].sum(axis=0)
            source_counts_2d += w[0].size

            for b in range(nbin_lens):
                w = np.where(lens_tomo_data['lens_bin']==b)
                # Summ all the PDFs from that bin
                lens_pdfs[b] += pz_data['pdf'][w].sum(axis=0)
                lens_counts[b] += w[0].size


        # Collect together the results from the different processors,
        # if we are running in parallel
        if self.comm:
            source_pdfs      = self.reduce(source_pdfs)
            source_pdfs_2d   = self.reduce(source_pdfs_2d)
            source_counts    = self.reduce(source_counts)
            source_counts_2d = self.reduce(source_counts_2d)
            lens_pdfs        = self.reduce(lens_pdfs)
            lens_counts      = self.reduce(lens_counts)

        if self.rank==0:
            # Normalize the stacks
            for b in range(nbin_source):
                source_pdfs[b] /= source_counts[b]
            source_pdfs_2d /= source_counts_2d


            # And finally save the outputs
            f = self.open_output("shear_photoz_stack")        
            self.save_result(f, "source", nbin_source, z, source_pdfs, source_counts)
            self.save_result(f, "source2d", 1, z, [source_pdfs_2d], [source_counts_2d])
            f.close()

            for b in range(nbin_lens):
                lens_pdfs[b] /= lens_counts[b]

            # And finally save the outputs
            f = self.open_output("lens_photoz_stack")
            self.save_result(f, "lens", nbin_lens, z, lens_pdfs, lens_counts)
            f.close()

    def reduce(self, x):
        # For scalars (i.e. just the 2D source count for now)
        # we just sum over all processors using reduce
        # For vectors we use Reduce, which applies specifically
        # to numpy arrays
        if np.isscalar(x):
            y = self.comm.reduce(x)
        else:
            y = np.zeros_like(x) if self.rank==0 else None
            self.comm.Reduce(x,y)
        return y


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

        # Save again but for the number of bins in the tomography catalog
        shear_tomo_file = self.open_input('shear_tomography_catalog')
        nbin_source = shear_tomo_file['tomography'].attrs['nbin_source']
        shear_tomo_file.close()

        lens_tomo_file = self.open_input('lens_tomography_catalog')
        nbin_lens = lens_tomo_file['tomography'].attrs['nbin_lens']
        lens_tomo_file.close()

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



class TXPhotozPlots(PipelineStage):
    """
    Make n(z) plots

    """
    name='TXPhotozPlots'
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
        plt.legend()
        plt.title("Lens n(z)")
        out1.close()

        f = self.open_input('shear_photoz_stack', wrapper=True)
        out2 = self.open_output('nz_source', wrapper=True)
        f.plot('source')
        plt.legend()
        plt.title("Source n(z)")
        out2.close()


class TXTrueNumberDensity(TXPhotozStack):
    """
    Fake an n(z) by histogramming the true redshift values for each object.
    """
    name='TXTrueNumberDensity'
    inputs = [
        ('shear_photometry_catalog', HDFFile),
        ('shear_tomography_catalog', TomographyCatalog),
        ('lens_photometry_catalog', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog)
    ]
    outputs = [
        ('shear_photoz_stack', NOfZFile),
        ('lens_photoz_stack', NOfZFile)
    ]
    config_options = {
        'chunk_rows': 5000,  # number of rows to read at once
        'zmax': float,
        'nz': int,
    }


    def run(self):
        # Set up the array we will stack the PDFs into
        # first get the sizes from metadata
        nbin_source, nbin_lens = self.get_metadata()

        # nz is number of bins in the code, but number of edges
        # in the input file for consistency with other stages
        nz = self.config['nz'] - 1
        zmax = self.config['zmax']

        # Space to accumulate histograms
        source_pdfs = np.zeros((nbin_source, nz))
        source_pdfs_2d = np.zeros(nz)
        lens_pdfs = np.zeros((nbin_lens, nz))
        source_counts = np.zeros(nbin_source)
        source_counts_2d = 0
        lens_counts = np.zeros(nbin_lens)

        
        # lower edges
        zedge = np.histogram([], range=(0,zmax), bins=nz)[1][:-1]

        # Data we need - the photometry catalog for DC2 has the true redshift
        # value in, and the tomography catalog has the binning.
        photo_iterator = self.iterate_hdf(
            'shear_photometry_catalog', # tag of input file to iterate through
            'photometry', # data group within file to look at
            ['redshift_true'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )

        shear_tomography_iterator = self.iterate_hdf(
            'shear_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['source_bin'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )

        lens_tomography_iterator = self.iterate_hdf(
            'lens_tomography_catalog', # tag of input file to iterate through
            'tomography', # data group within file to look at
            ['lens_bin'], # column(s) to read
            self.config['chunk_rows']  # number of rows to read at once
        )

        warnings.warn("WEIGHTS/RESPONSE ARE NOT CURRENTLY INCLUDED CORRECTLY in PZ STACKING")

        # So we just do a single loop through the pair of files.
        for (_, _, pz_data), (s1, e1, shear_tomo_data), (s2, e2, lens_tomo_data) in zip(photo_iterator, shear_tomography_iterator, lens_tomography_iterator):
            # pz_data and tomo_data are dictionaries with the keys as column names and the 
            # values as numpy arrays with a chunk of data (chunk_rows long) in.
            # Each iteration through the loop we get a new chunk.
            print(f"Process {self.rank} read data chunk {s1:,} - {e1:,}")
            # The method also yields the start and end positions in the file.  We don't need those
            # here because we are just summing them all together.  That's what the underscores
            # are above.

            z = pz_data['redshift_true']

            # Now for each tomographic bin find all the objects in that bin.
            # There is probably a better way of doing this.
            for b in range(nbin_source):
                # Locate objects in this bin
                w = np.where(shear_tomo_data['source_bin']==b)
                # Accumulate the histogram and the count for this bin
                source_pdfs[b] += np.histogram(z[w], bins=nz, range=(0,zmax))[0]
                source_counts[b] += w[0].size

            # For the 2D source bin we take every object that is selected
            # for any tomographic bin (the non-selected objects
            # have bin=-1)s
            w = np.where(shear_tomo_data['source_bin']>=0)
            source_pdfs_2d +=  np.histogram(z[w], bins=nz, range=(0,zmax))[0]
            source_counts_2d += w[0].size

            for b in range(nbin_lens):
                w = np.where(lens_tomo_data['lens_bin']==b)
                print(z[w])
                lens_pdfs[b] +=  np.histogram(z[w], bins=nz, range=(0,zmax))[0]
                lens_counts[b] += w[0].size

        # Collect together the results from the different processors,
        # if we are running in parallel.
        # Though it's barely worth it for this as it's so fast.
        if self.comm:
            source_pdfs      = self.reduce(source_pdfs)
            source_pdfs_2d   = self.reduce(source_pdfs_2d)
            source_counts    = self.reduce(source_counts)
            source_counts_2d = self.reduce(source_counts_2d)
            lens_pdfs        = self.reduce(lens_pdfs)
            lens_counts      = self.reduce(lens_counts)

        # Only the root process saves the data
        if self.rank==0:
            # Normalize the stacks
            for b in range(nbin_source):
                source_pdfs[b] /= source_counts[b]
            source_pdfs_2d /= source_counts_2d

            # And finally save the outputs
            f = self.open_output("shear_photoz_stack")
            # These are inherited from the parent class.      
            self.save_result(f, "source", nbin_source, zedge, source_pdfs, source_counts)
            self.save_result(f, "source2d", 1, zedge, [source_pdfs_2d], [source_counts_2d])
            f.close()

            # Normalize the stacks
            for b in range(nbin_lens):
                print(b, lens_counts[b])
                lens_pdfs[b] /= lens_counts[b]

            # And finally save the outputs
            f = self.open_output("lens_photoz_stack")
            # These are inherited from the parent class.
            self.save_result(f, "lens", nbin_lens, zedge, lens_pdfs, lens_counts)
            f.close()

    def get_metadata(self):
        """
        Load the number of bins and also check that the photometry file
        has a redshift column in it.

        Returns
        -------
        nbin_source: int

        nbin_lens: int

        """

        # Check we are running on a photo file with redshift_true
        photo_file = self.open_input('shear_photometry_catalog')
        has_z = 'redshift_true' in photo_file['photometry'].keys()
        photo_file.close()
        if not has_z:
            msg = ("The photometry_catalog file you supplied does not have a redshift_true column. "
                   "If you're running on sims you need to make sure to ingest that column from GCR. "
                   "If you're running on real data then sadly this isn't going to work. "
                   "Use a different stacking stage."
                )
            raise ValueError(msg)

        # Save again but for the number of bins in the tomography catalog
        shear_tomo_file = self.open_input('shear_tomography_catalog')
        nbin_source = shear_tomo_file['tomography'].attrs['nbin_source']
        shear_tomo_file.close()

        shear_tomo_file = self.open_input('lens_tomography_catalog')
        nbin_lens = lens_tomo_file['tomography'].attrs['nbin_lens']
        lens_tomo_file.close()

        return nbin_source, nbin_lens




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



