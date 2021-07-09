from .base_stage import PipelineStage
from .data_types import FitsFile, TomographyCatalog, HDFFile, PNGFile, NOfZFile
from .utils.mpi_utils import in_place_reduce
import numpy as np
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TXPhotozSourceDIR(PipelineStage):
    pass
    """
    Naively stack photo-z PDFs in bins according to previous selections.

    This parent class does only the source bins.
    """
    name='TXPhotozSourceDIR'
    inputs = [
        ('cosmos_weights', FitsFile),
        ('shear_tomography_catalog', TomographyCatalog),
    ]
    outputs = [
        ('shear_photoz_dir', NOfZFile),
    ]
    config_options = {'pz_code': 'ephor_ab',
                      'pz_mark': 'best',
                      'pz_bins': [0.15, 0.50, 0.75, 1.00, 1.50],
                      'nz_bin_num': 200,
                      'nz_bin_max': 3.0
    }

    def run(self):
        """
        Run the analysis for this stage.
        
         - Get metadata and allocate space for output
         - Set up iterators to loop through tomography and PDF input files
         - Accumulate the PDFs for each object in each bin
         - Divide by the counts to get the stacked PDF
        """

        logger.info("Getting COSMOS N(z)s.")
        pzs_cosmos = self.get_nz_cosmos()

        # Save the stacks
        f = self.open_output("shear_photoz_dir")
        self.save('source', pzs_cosmos, f)
        f.close()

    def get_nz_cosmos(self):
        """
        Get N(z) from weighted COSMOS-30band data
        """
        zi_arr = self.config['pz_bins'][:-1]
        zf_arr = self.config['pz_bins'][1:]

        if self.config['pz_code'] == 'ephor_ab':
            pz_code = 'eab'
        elif self.config['pz_code'] == 'frankenz':
            pz_code = 'frz'
        elif self.config['pz_code'] == 'nnpz':
            pz_code = 'nnz'
        else:
            raise KeyError("Photo-z method "+self.config['pz_code'] +
                           " unavailable. Choose ephor_ab, frankenz or nnpz")

        if self.config['pz_mark'] not in ['best', 'mean', 'mode', 'mc']:
            raise KeyError("Photo-z mark "+self.config['pz_mark'] +
                           " unavailable. Choose between best, mean, " +
                           "mode and mc")

        self.column_mark = 'pz_'+self.config['pz_mark']+'_'+pz_code

        weights_file = fits.open(self.get_input('cosmos_weights'))[1].data

        pzs = []
        for zi, zf in zip(zi_arr, zf_arr):
            msk_cosmos = ((weights_file[self.column_mark] <= zf) &
                          (weights_file[self.column_mark] > zi))
            hz, bz = np.histogram(weights_file[msk_cosmos]['PHOTOZ'],
                                  bins=self.config['nz_bin_num'],
                                  range=[0., self.config['nz_bin_max']],
                                  weights=weights_file[msk_cosmos]['weight'])
            hnz, bnz = np.histogram(weights_file[msk_cosmos]['PHOTOZ'],
                                    bins=self.config['nz_bin_num'],
                                    range=[0., self.config['nz_bin_max']])
            ehz = np.zeros(len(hnz))
            ehz[hnz > 0] = (hz[hnz > 0]+0.)/np.sqrt(hnz[hnz > 0]+0.)
            pzs.append([bz[:-1], bz[1:], hnz, (hz+0.)/np.sum(hz+0.), ehz])
        return np.array(pzs)

    def save(self, name, pzs, outfile):
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

        logger.info('Writing photoz_dir for {} to {}. '.format(name, self.get_output('shear_photoz_dir')))

        # Create a group inside for the n_of_z data we made here.
        group = outfile.create_group(f"n_of_z/{name}")

        # HDF has "attributes" which are for small metadata like this
        nbin = len(self.config['pz_bins']) - 1
        z_grid = 0.5 * (pzs[0, 0, :] + pzs[0, 1, :])
        group.attrs["nbin"] = nbin
        group.attrs["nz"] = len(z_grid)

        # Save the redshift sampling.
        group.create_dataset("z", data=z_grid)

        # And all the bins separately
        for b in range(nbin):
            group.attrs[f"z_i_{b}"] = pzs[b, 0, :]
            group.attrs[f"z_f_{b}"] = pzs[b, 1, :]
            group.attrs[f"count_{b}"] = pzs[b, 2, :]
            group.attrs[f"bin_{b}"] = pzs[b, 3, :]
            group.attrs[f"enz_cosmos_{b}"] = pzs[b, 4, :]

class TXPhotozDIR(TXPhotozSourceDIR):
    """
    This PZ stacking subclass does both source and lens

    """
    name='TXPhotozStack'

    inputs = [
        ('cosmos_weights', FitsFile),
        ('shear_tomography_catalog', TomographyCatalog),
        ('lens_tomography_catalog', TomographyCatalog)
    ]
    outputs = [
        ('shear_photoz_dir', NOfZFile),
        ('lens_photoz_dir', NOfZFile)
    ]

    def run(self):
        """
        Run the analysis for this stage.

         - Get metadata and allocate space for output
         - Set up iterators to loop through tomography and PDF input files
         - Accumulate the PDFs for each object in each bin
         - Divide by the counts to get the stacked PDF
        """

        logger.info("Getting COSMOS N(z)s.")
        pzs_cosmos = self.get_nz_cosmos()

        # Save the stacks
        f = self.open_output("shear_photoz_stack")
        self.save('source', pzs_cosmos, f)
        f.close()
        logger.info('This is a placeholder for when we get the COSMOS shear weights.')
        f = self.open_output("lens_photoz_stack")
        self.save('lens', pzs_cosmos, f)
        f.close()


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


class TXTrueNumberDensity(TXPhotozStack):
    """
    Fake an n(z) by histogramming the true redshift values for each object.
    Uses the same method as its parent but loads the data
    differently and uses the truth redshift as a delta function PDF
    """
    name='TXTrueNumberDensity'
    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_tomography_catalog', TomographyCatalog),
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


    def data_iterator(self):
        return self.combined_iterators(
                self.config['chunk_rows'],
                'photometry_catalog', # tag of input file to iterate through
                'photometry', # data group within file to look at
                ['redshift_true'], # column(s) to read

                'shear_tomography_catalog', # tag of input file to iterate through
                'tomography', # data group within file to look at
                ['source_bin'], # column(s) to read

                'lens_tomography_catalog', # tag of input file to iterate through
                'tomography', # data group within file to look at
                ['lens_bin'], # column(s) to read
            )

    def stack_data(self, data, outputs):
        source_stack, source2d_stack, lens_stack = outputs
        source_stack.add_delta_function(data['source_bin'], data['redshift_true'])
        bin2d = data['source_bin'].clip(-1, 0)
        source2d_stack.add_delta_function(bin2d, data['redshift_true'])
        lens_stack.add_delta_function(data['lens_bin'], data['redshift_true'])


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

        lens_tomo_file = self.open_input('lens_tomography_catalog')
        nbin_lens = lens_tomo_file['tomography'].attrs['nbin_lens']
        lens_tomo_file.close()

        return z, nbin_source, nbin_lens


class TXSourceTrueNumberDensity(TXPhotozSourceStack):
    """
    Fake an n(z) by histogramming the true redshift values for each object.
    Uses the same method as its parent but loads the data
    differently and uses the truth redshift as a delta function PDF
    """
    name='TXSourceTrueNumberDensity'
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
