from ceci import PipelineStage
from .data_types import FitsFile
from astropy.table import Table, hstack
import numpy as np
# from .plot_utils import plot_histo
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spatial
from .lens_selector_hsc import TXHSCLensSelector


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TXCOSMOSWeight(TXHSCLensSelector):
    name = "TXCOSMOSWeight"
    inputs = [('cosmos_data', FitsFile),
              ('cosmos_hsc', FitsFile),
              ('cosmos_hsc_source_weights', FitsFile)]
    outputs = [('cosmos_photo_weights', FitsFile),
               ('cosmos_source_weights', FitsFile)]
    config_options = {'mag_i_cut': 24.5,
                      'n_neighbors': 10}

    def run(self):
        """
        Main function.
        This stage matches the COSMOS-30band data with the HSC COSMOS sample
        cut with the same criteria as our data and produces colour-space
        weights to match our sample so it can be used to estimate redshift
        distributions.
        """

        # Read HSC COSMOS catalog
        logger.info("Reading HSC COSMOS catalog.")
        cat_photo = Table.read(self.get_input('cosmos_hsc'))
        # Clean nulls and nans
        sel = np.ones(len(cat_photo), dtype=bool)
        isnull_names = []
        for key in cat_photo.keys():
            if key.__contains__('isnull'):
                sel[cat_photo[key]] = 0
                isnull_names.append(key)
            else:
                # Keep photo-z's even if they're NaNs
                if not key.startswith("pz_"):
                    sel[np.isnan(cat_photo[key])] = 0
        cat_photo.remove_columns(isnull_names)
        cat_photo.remove_rows(~sel)
        # Select lens objects
        lens_cat = self.select_lens(cat_photo)
        cat_photo = cat_photo[lens_cat]

        # Define shear catalog
        cat_photo['shear_cat'] = self.shear_cut(cat_photo)

        ####
        # Read COSMOS-30band
        logger.info("Reading COSMOS 30band catalog.")
        cat30 = fits.open(self.get_input('cosmos_data'))[1].data
        lim_indices = np.where((0.01 < cat30['PHOTOZ']) &
                               (9 > cat30['PHOTOZ']) &
                               (cat30['TYPE'] == 0) &
                               (cat30['ZP_2'] < 0) &
                               (cat30['MASS_BEST'] > 7.5) &
                               (np.maximum(cat30['ZPDF_H68']-cat30['ZPDF'],
                                           cat30['ZPDF']-cat30['ZPDF_L68']) <
                                0.05*(1+cat30['PHOTOZ'])) &
                               (cat30['CHI2_BEST'] < cat30['CHIS']) &
                               (cat30['CHI2_BEST']/cat30['NBFILT'] < 5.))
        cat30 = cat30[lim_indices]

        ####
        # Match coordinates
        logger.info("Matching coordinates.")
        cosmos_crd = SkyCoord(ra=np.array(cat30['ALPHA_J2000'])*u.deg,
                              dec=np.array(cat30['DELTA_J2000'])*u.deg)
        hsc_crd = SkyCoord(ra=np.array(cat_photo['ra'])*u.deg,
                           dec=np.array(cat_photo['dec'])*u.deg)
        # Nearest neighbors
        cosmos_index, dist_2d, dist_3d = hsc_crd.match_to_catalog_sky(cosmos_crd)
        # Cut everything further than 1 arcsec
        mask = dist_2d.degree*60*60 < 1
        cat_photo_good = cat_photo[mask]
        cat30_good = cat30[cosmos_index[mask]]
        cosmos_photo_index_matched = cosmos_index[mask]

        t1 = Table.from_pandas(pd.DataFrame(cat30_good))
        keys_t2 = ['gcmodel_mag', 'rcmodel_mag', 'icmodel_mag', 'zcmodel_mag',
                   'ycmodel_mag', 'pz_mean_eab', 'pz_mode_eab', 'pz_best_eab',
                   'pz_mc_eab', 'shear_cat']
        t2 = Table.from_pandas(pd.DataFrame(np.transpose([np.array(cat_photo_good[k])
                                                          for k in keys_t2]),
                                            index=range(len(cat_photo_good)),
                                            columns=keys_t2))
        cat_photo_matched = hstack([t1, t2])

        ####
        # Get color-space weights
        weights_dir_photo = self.get_weights(cat_photo, cat_photo_matched)

        cat_source = cat_photo[cat_photo['shear_cat']]
        cat_source_matched = cat_photo_matched[cat_photo_matched['shear_cat']]
        cosmos_source_index_matched = cosmos_photo_index_matched[cat_photo_matched['shear_cat']]
        weights_dir_source = self.get_weights(cat_source, cat_source_matched)
        weights_source, cat_source_index = self.get_source_weights(cat_source_matched)
        cat_source_matched = cat_source_matched[cat_source_index]
        cosmos_source_index_matched = cosmos_source_index_matched[cat_source_index]

        ####
        # Write output
        logger.info('Writing weights for photo catalog.')
        keys_t1 = ['ALPHA_J2000', 'DELTA_J2000', 'gcmodel_mag', 'rcmodel_mag',
                   'icmodel_mag', 'zcmodel_mag', 'ycmodel_mag',
                   'pz_best_eab', 'PHOTOZ', 'MNUV', 'MU', 'MB',
                   'MV', 'MR', 'MI', 'MZ', 'MY', 'MJ', 'MH', 'MK']
        t1 = Table.from_pandas(pd.DataFrame(np.transpose([cat_photo_matched[k]
                                                          for k in keys_t1]),
                                            columns=keys_t1))
        t2 = Table.from_pandas(pd.DataFrame(np.transpose(weights_dir_photo),
                                            columns=['weight']))
        t3 = Table.from_pandas(pd.DataFrame(np.transpose(cosmos_photo_index_matched),
                                            columns=['cosmos_index_matched']))
        cat_photo_weights = hstack([t1, t2, t3])
        cat_photo_weights.write(self.get_output('cosmos_photo_weights'),
                          overwrite=True)

        logger.info('Writing weights for shear catalog.')
        t1 = Table.from_pandas(pd.DataFrame(np.transpose([cat_source_matched[k]
                                                          for k in keys_t1]),
                                            columns=keys_t1))
        t2 = Table.from_pandas(pd.DataFrame(np.transpose(weights_source*weights_dir_source),
                                            columns=['weight']))
        t3 = Table.from_pandas(pd.DataFrame(np.transpose(cosmos_source_index_matched),
                                            columns=['cosmos_index_matched']))
        cat_weights = hstack([t1, t2, t3])
        cat_weights.write(self.get_output('cosmos_source_weights'),
                          overwrite=True)

        # # Plot magnitude distribution
        # for b in ['g', 'r', 'i', 'z', 'y']:
        #     plot_histo(self.config, '%s_mag_COSMOS' % b,
        #                [cat_weights['%scmodel_mag' % b],
        #                 cat['%scmodel_mag' % b]],
        #                ['COSMOS', 'HSC'],
        #                weights=[cat_weights['weight'],
        #                         np.ones(len(cat))],
        #                density=True, logy=True, bins=50)
        # # Plot weights distribution
        # plot_histo(self.config, 'COSMOS_weights',
        #            [cat_weights['weight']], ['weights'],
        #            density=True, logy=True, bins=50)

        # Permissions on NERSC
        # os.system('find /global/cscratch1/sd/damonge/GSKY/ -type d -exec chmod -f 777 {} \;')
        # os.system('find /global/cscratch1/sd/damonge/GSKY/ -type f -exec chmod -f 666 {} \;')

    def get_weights(self, cat, cat_matched):
        """
        Get DIR weights.
        Parameters
        ----------
        cat
        cat_matched

        Returns
        -------

        """

        bands = ['g', 'r', 'i', 'z', 'y']
        logger.info("Computing color-space weights.")
        train_sample = np.transpose(np.array([np.array(cat_matched['%scmodel_mag' % m])
                                              for m in bands]))
        photoz_sample = np.transpose(np.array([np.array(cat['%scmodel_mag' % m])
                                               for m in bands]))

        # Find nearest neighbors in color space
        n_nbrs = NearestNeighbors(n_neighbors=self.config['n_neighbors'],
                                  algorithm='kd_tree',
                                  metric='euclidean').fit(train_sample)
        distances, _ = n_nbrs.kneighbors(train_sample)
        # Get maximum distance
        distances = np.amax(distances, axis=1)
        # Find all photo-z objects within this maximum distance
        # for each COSMOS object
        tree_NN_lookup = spatial.cKDTree(photoz_sample, leafsize=40)
        num_photoz = np.array([len(tree_NN_lookup.query_ball_point(t, d+1E-6))
                               for t, d in zip(train_sample, distances)])
        # Weights are ratio of number of photo-z neighbors to
        # COSMOS neighbors (normalized by the number of photo-z objects)
        weights = np.true_divide(num_photoz*len(train_sample),
                                 self.config['n_neighbors'] *
                                 len(photoz_sample))
        logger.info('Sum of COSMOS weights = {}.', np.sum(weights))

        return weights

    def shear_cut(self, cat):
        """
        Apply additional shear cuts to catalog.
        :param cat:
        :return:
        """

        logger.info('Applying shear cuts to catalog.')

        ishape_flags_mask = ~cat['ishape_hsm_regauss_flags']
        ishape_sigma_mask = ~np.isnan(cat['ishape_hsm_regauss_sigma'])
        ishape_resolution_mask = cat['ishape_hsm_regauss_resolution'] >= 0.3
        ishape_shear_mod_mask = (cat['ishape_hsm_regauss_e1'] ** 2 + cat['ishape_hsm_regauss_e2'] ** 2) < 2
        ishape_sigma_mask *= (cat['ishape_hsm_regauss_sigma'] >= 0.) * (cat['ishape_hsm_regauss_sigma'] <= 0.4)
        # Remove masked objects
        if self.config['mask_type'] == 'arcturus':
            star_mask = cat['mask_Arcturus'].astype(bool)
        elif self.config['mask_type'] == 'sirius':
            star_mask = np.logical_not(cat['iflags_pixel_bright_object_center'])
            star_mask *= np.logical_not(cat['iflags_pixel_bright_object_any'])
        else:
            raise KeyError("Mask type " + self.config['mask_type'] +
                           " not supported. Choose arcturus or sirius")
        fdfc_mask = cat['wl_fulldepth_fullcolor']

        shearmask = ishape_flags_mask * ishape_sigma_mask * ishape_resolution_mask * ishape_shear_mod_mask * star_mask * fdfc_mask

        return shearmask

    def get_source_weights(self, cat_source):
        """
        Get the HSC source weights from file and cross-match to catalog.
        Parameters
        ----------
        cat_source

        Returns
        -------

        """

        logger.info("Getting shear weights.")
        cat_source_weights = Table.read(self.get_input('cosmos_hsc_source_weights'))

        cat_source_crd = SkyCoord(ra=np.array(cat_source['ALPHA_J2000'])*u.deg,
                              dec=np.array(cat_source['DELTA_J2000'])*u.deg)
        cat_source_weights_crd = SkyCoord(ra=np.array(cat_source_weights['ra'])*u.deg,
                           dec=np.array(cat_source_weights['dec'])*u.deg)
        # Nearest neighbors
        cat_source_index, dist_2d, dist_3d = cat_source_weights_crd.match_to_catalog_sky(cat_source_crd)
        # Cut everything further than 1 arcsec
        mask = dist_2d.degree*60*60 < 1
        cat_source_weights_good = cat_source_weights[mask]
        cat_source_good = cat_source[cat_source_index[mask]]
        cat_source_index = cat_source_index[mask]

        t1 = Table.from_pandas(pd.DataFrame(cat_source_weights_good))
        keys_t2 = ['gcmodel_mag', 'rcmodel_mag', 'icmodel_mag', 'zcmodel_mag',
                   'ycmodel_mag', 'pz_mean_eab', 'pz_mode_eab', 'pz_best_eab',
                   'pz_mc_eab', 'shear_cat']
        t2 = Table.from_pandas(pd.DataFrame(np.transpose([np.array(cat_source_good[k])
                                                          for k in keys_t2]),
                                            index=range(len(cat_source_good)),
                                            columns=keys_t2))
        cat_source_matched = hstack([t1, t2])

        return cat_source_matched, cat_source_index

if __name__ == '__main__':
    cls = PipelineStage.main()
