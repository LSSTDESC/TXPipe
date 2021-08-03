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
        ('cosmos_source_weights', FitsFile),
    ]
    outputs = [
        ('shear_photoz_dir', NOfZFile),
    ]
    config_options = {'pz_code': 'ephor_ab',
                      'pz_mark': 'best',
                      'source_zbin_edges': [0.3, 0.6, 0.9, 1.2, 1.5],
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
        pzs_cosmos = self.get_nz_cosmos('source')

        # Save the stacks
        f = self.open_output("shear_photoz_dir")
        self.save('source', pzs_cosmos, f)
        f.close()

    def get_nz_cosmos(self, name):
        """
        Get N(z) from weighted COSMOS-30band data
        """

        if name == 'source':
            zi_arr = self.config['source_zbin_edges'][:-1]
            zf_arr = self.config['source_zbin_edges'][1:]
        elif name == 'lens':
            zi_arr = self.config['lens_zbin_edges'][:-1]
            zf_arr = self.config['lens_zbin_edges'][1:]
        else:
            raise NotImplementedError('Only name = source, lens supported. Aborting.')

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

        if name == 'source':
            weights_file = fits.open(self.get_input('cosmos_source_weights'))[1].data
        elif name == 'lens':
            weights_file = fits.open(self.get_input('cosmos_photo_weights'))[1].data
        else:
            raise NotImplementedError('Only name = source, lens supported. Aborting.')

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
    name='TXPhotozDIR'

    inputs = [
        ('cosmos_photo_weights', FitsFile),
        ('cosmos_source_weights', FitsFile),
    ]
    outputs = [
        ('shear_photoz_dir', NOfZFile),
        ('lens_photoz_dir', NOfZFile)
    ]
    config_options = {'pz_code': 'ephor_ab',
                      'pz_mark': 'best',
                      'source_zbin_edges': [0.3, 0.6, 0.9, 1.2, 1.5],
                      'lens_zbin_edges': [0.15, 0.50, 0.75, 1.00, 1.50],
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

        logger.info("Getting COSMOS N(z)s for sources.")
        pzs_cosmos_source = self.get_nz_cosmos('source')
        logger.info("Getting COSMOS N(z)s for lenses.")
        pzs_cosmos_lens = self.get_nz_cosmos('lens')

        # Save the stacks
        logger.info('This is a placeholder for when we get the COSMOS shear weights.')
        f = self.open_output("shear_photoz_dir")
        self.save('source', pzs_cosmos_source, f)
        f.close()
        f = self.open_output("lens_photoz_dir")
        self.save('lens', pzs_cosmos_lens, f)
        f.close()



