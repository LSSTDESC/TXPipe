from ..base_stage import PipelineStage
from ..twopoint import TXTwoPoint 
from ..data_types import HDFFile, ShearCatalog, TomographyCatalog, RandomsCatalog, FiducialCosmology, SACCFile, PhotozPDFFile, PNGFile, TextFile
from ..utils.calibration_tools import apply_metacal_response, apply_lensfit_calibration 
from ..utils.calibration_tools import read_shear_catalog_type
import numpy as np
import random
import collections
import sys

# This creates a little mini-type, like a struct,
# for holding individual measurements
Measurement = collections.namedtuple(
    'Measurement',
    ['corr_type', 'object', 'i', 'j'])

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2
SHEAR_POS_SELECT = 3

class TXSelfCalibrationIA(TXTwoPoint):
    """
    This is the subclass of the Twopoint class that will handle calculating the
    correlations needed for doing the Self-calibration Intrinsic alignment
    estimation. 

    It requires estimating 3d two-point correlations. We calculate the 
    galaxy - galaxy lensing auto-correlation in each source bin.
    We do this twice, once we add a selection-function, such that we only selects 
    the pairs where we have the shear object in front of the object used for density. 
    Once without imposing this selection funciton. 
    """
    name = 'TXSCIA'
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats_source', RandomsCatalog),
        ('patch_centers', TextFile),
        ('photoz_pdfs', PhotozPDFFile),
        ('fiducial_cosmology', FiducialCosmology),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('twopoint_data_SCIA', SACCFile),
        ('gammaX_scia', SACCFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':250.,
        'nbins':20,
        'bin_slop':0.1,
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'source_bins':[-1],
        'lens_bins':[-1],
        'reduce_randoms_size':1.0,
        'do_shear_pos': True,
        'do_pos_pos': False,
        'do_shear_shear': False, 
        'var_method': 'jackknife',
        '3Dcoords': True,
        'metric': 'Rperp',
        'use_true_shear': False,
        'subtract_mean_shear':False,
        'redshift_shearcatalog': False,
        }

    def run(self):

        super().run()

    def select_calculations(self, data):
        source_list = data['source_list']
        calcs = []
        
        if self.config['do_shear_pos']:
            k = SHEAR_POS
            l = SHEAR_POS_SELECT
            for i in source_list:
                calcs.append((i,i,k))
                calcs.append((i,i,l))
        
        if self.config['do_pos_pos']:
            if not 'random_bin' in data:
                raise ValueError('You need to have a random catalog to calculate position-position correlations')
            k = POS_POS
            for i in source_list:
                calcs.append((i,i,k))
        
        if self.config['do_shear_shear']:
            k = SHEAR_SHEAR
            for i in source_list:
                for j in range(i+1):
                    if j in source_list:
                        calcs.append((i,j,k))


        if self.rank==0:
            print(f"Running these calculations: {calcs}")
        
        return calcs

    def load_shear_catalog(self, data):

        # Columns we need from the shear catalog
        read_shear_catalog_type(self)

        if self.config['shear_catalog_type']=='metacal':
            if self.config['use_true_shear']:
                cat_cols = ['ra', 'dec', 'true_g1', 'true_g2', 'mcal_flags']
            else:
                cat_cols = ['ra', 'dec', 'mcal_g1', 'mcal_g2', 'mcal_flags']
                
        else:
            cat_cols = ['ra', 'dec', 'g1', 'g2', 'weight','flags','sigma_e','m']
        print(f"Loading shear catalog columns: {cat_cols}")

        f = self.open_input('shear_catalog')
        g = f['shear']
        for col in cat_cols:
            print(f"Loading {col}")
            data[col] = g[col][:]

        if self.config['3Dcoords']:
            #Temporary fix for not running PDF's on DES
            if self.config['redshift_shearcatalog']:
                data['mu'] = g['mean_z']
            else: 
                h = self.open_input('photoz_pdfs')
                g = h['point_estimates']
                data['mu'] = g['z_mean'][:]

    def load_random_catalog(self, data):
        # For now we are just bypassing this, since it is not needed

        filename = self.get_input('random_cats_source')
        if filename =='None':
            filename = None

        if filename is None:
            print("Not using randoms")
            return

        
        # Columns we need from the tomography catalog
        randoms_cols = ['dec','ra','bin','z']
        print(f"Loading random catalog columns: {randoms_cols}")

        f = self.open_input('random_cats_source')
        group = f['randoms']

        cut = self.config['reduce_randoms_size']
        if 0.0<cut<1.0:
            N = group['dec'].size
            sel = np.random.uniform(size=N) < cut
        else:
            sel = slice(None)

        data['random_ra'] =  group['ra'][sel]
        data['random_dec'] = group['dec'][sel]

        data['random_bin'] = group['bin'][sel]
        data['random_z'] = group['z'][sel]

        f.close()


    def get_lens_catalog(self, data, meta, i):
        import treecorr
        import pyccl as ccl

        #Note that we are here actually loading the source bin as the lens bin!
        #mask = data['source_bin'] == i
        g1, g2, mask = self.get_calibrated_catalog_bin(data, meta, i)

        
        if 'lens_ra' in data:
            ra = data['lens_ra'][mask]
            dec = data['lens_dec'][mask]
        else:
            ra = data['ra'][mask]
            dec = data['dec'][mask]
        
        if self.config['3Dcoords']:
            mu = data['mu'][mask]
            cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology'))
            r = ccl.background.comoving_radial_distance(cosmo, 1/(1+mu))


            if self.config['var_method']=='jackknife':
                patch_centers = self.get_input('patch_centers')
                cat = treecorr.Catalog(
                    ra=ra, dec=dec, r= r,
                    ra_units='degree', dec_units='degree',
                    patch_centers=patch_centers)
            else:
                cat = treecorr.Catalog(
                    ra=ra, dec=dec, r=r,
                    ra_units='degree', dec_units='degree')

            if 'random_bin' in data:
                random_mask = data['random_bin']==i
                z_rand = data['random_z'][random_mask]
                r_rand = ccl.background.comoving_radial_distance(cosmo, 1/(1+z_rand))
                if self.config['var_method']=='jackknife':
                    rancat  = treecorr.Catalog(
                        ra=data['random_ra'][random_mask], dec=data['random_dec'][random_mask],
                        r = r_rand, ra_units='degree', dec_units='degree',
                        patch_centers=patch_centers)
                else:
                    rancat  = treecorr.Catalog(
                        ra=data['random_ra'][random_mask], dec=data['random_dec'][random_mask],
                        r = r_rand, ra_units='degree', dec_units='degree')
            else:
                rancat = None
        else:
            if self.config['var_method']=='jackknife':
                patch_centers = self.get_input('patch_centers')
                cat = treecorr.Catalog(
                    ra=ra, dec=dec,
                    ra_units='degree', dec_units='degree',
                    patch_centers=patch_centers)
            else:
                cat = treecorr.Catalog(
                    ra=ra, dec=dec,
                    ra_units='degree', dec_units='degree')

            if 'random_bin' in data:
                random_mask = data['random_bin']==i
                if self.config['var_method']=='jackknife':
                    rancat  = treecorr.Catalog(
                        ra=data['random_ra'][random_mask], dec=data['random_dec'][random_mask],
                        ra_units='degree', dec_units='degree',
                        patch_centers=patch_centers)
                else:
                    rancat  = treecorr.Catalog(
                        ra=data['random_ra'][random_mask], dec=data['random_dec'][random_mask],
                        ra_units='degree', dec_units='degree')
            else:
                rancat = None

        return cat, rancat

    def get_shear_catalog(self, data, meta, i):
        import treecorr
        import pyccl as ccl
        g1, g2, mask = self.get_calibrated_catalog_bin(data, meta, i)
        
        if self.config['3Dcoords']:
            mu = data['mu'][mask]
            cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology'))
            r = ccl.background.comoving_radial_distance(cosmo, 1/(1+mu))

            if self.config['var_method']=='jackknife':
                patch_centers = self.get_input('patch_centers')
                cat = treecorr.Catalog(
                    g1 = g1,
                    g2 = g2,
                    r = r,
                    ra = data['ra'][mask],
                    dec = data['dec'][mask],
                    ra_units='degree', dec_units='degree',
                    patch_centers=patch_centers)
                    #npatch=self.config['npatch'])
            else:
                cat = treecorr.Catalog(
                    g1 = g1,
                    g2 = g2,
                    r = r,
                    ra = data['ra'][mask],
                    dec = data['dec'][mask],
                    ra_units='degree', dec_units='degree')
        else:
            if self.config['var_method']=='jackknife':
                patch_centers = self.get_input('patch_centers')
                cat = treecorr.Catalog(
                    g1 = g1,
                    g2 = g2,
                    ra = data['ra'][mask],
                    dec = data['dec'][mask],
                    ra_units='degree', dec_units='degree',
                    patch_centers=patch_centers)
                    #npatch=self.config['npatch'])
            else:
                cat = treecorr.Catalog(
                    g1 = g1,
                    g2 = g2,
                    ra = data['ra'][mask],
                    dec = data['dec'][mask],
                    ra_units='degree', dec_units='degree')

        return cat

    def calculate_shear_pos_select(self,data, meta, i, j):
        # This is the added calculation that uses the selection function, as defined in our paper. it picks
        # out all the pairs where the object in the source catalog is in front of the lens object. 
        # note we are looking at auto-correlations for the source bins!
        import treecorr 

        cat_i = self.get_shear_catalog(data, meta, i)
        n_i = cat_i.nobj

        cat_j, rancat_j = self.get_lens_catalog(data, meta, j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        print(f"Rank {self.rank} calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")

        ng = treecorr.NGCorrelation(self.config, max_rpar = 0.0)
        ng.process(cat_j, cat_i)

        if rancat_j:
            rg = treecorr.NGCorrelation(self.config, max_rpar = 0.0)
            rg.process(rancat_j, cat_i)
        else:
            rg = None

        ng.calculateXi(rg=rg)

        return ng

    def calculate_shear_pos(self, data, meta, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(data, meta, i)
        n_i = cat_i.nobj

        cat_j, rancat_j = self.get_lens_catalog(data, meta, j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        print(f"Rank {self.rank} calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")

        ng = treecorr.NGCorrelation(self.config)
        ng.process(cat_j, cat_i)

        if rancat_j:
            rg = treecorr.NGCorrelation(self.config)
            rg.process(rancat_j, cat_i)
        else:
            rg = None

        ng.calculateXi(rg=rg)

        return ng

    def call_treecorr(self, data, meta, i, j, k):
        import sacc 

        if k==SHEAR_SHEAR:
            xx = self.calculate_shear_shear(data, meta, i, j)
            xtype = "combined"
        elif k==SHEAR_POS:
            xx = self.calculate_shear_pos(data, meta, i, j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k==SHEAR_POS_SELECT:
            xx = self.calculate_shear_pos_select(data, meta, i, j)
            xtype = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='ts')
        elif k==POS_POS:
            xx = self.calculate_pos_pos(data, meta, i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        else:
            raise ValueError(f"Unknown correlation function {k}")

        result = Measurement(xtype, xx, i, j)

        sys.stdout.flush()
        return result

    def write_output(self, data, meta, results):
        import sacc
        import treecorr
        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMATS = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='ts')
        WTHETA = sacc.standard_types.galaxy_density_xi
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        GAMMAXS = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='xs')

        S = sacc.Sacc()
        if self.config['do_shear_pos'] == True:
            S2 = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data
        f = self.open_input('shear_photoz_stack')

        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        for i in data['source_list']:
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            S.add_tracer('NZ', f'source_{i}', z, Nz)
            if self.config['do_shear_pos'] == True:
                S2.add_tracer('NZ', f'source_{i}', z, Nz)

        # f = self.open_input('lens_photoz_stack')
        # For both source and lens
        # for i in data['lens_list']:
        #    z = f['n_of_z/lens/z'][:]
        #    Nz = f[f'n_of_z/lens/bin_{i}'][:]
        #    S.add_tracer('NZ', f'lens_{i}', z, Nz)
        # Closing n(z) file
        f.close()

        # Now build up the collection of data points, adding them all to
        # the sacc data one by one.
        comb = []
        for d in results:
            # First the tracers and generic tags
            tracer1 = f'source_{d.i}' #if d.corr_type in [XI, GAMMAT,GAMMATS, ] else f'lens_{d.i}'
            tracer2 = f'source_{d.j}' #if d.corr_type in [XI, GAMMAT, GAMMATS] else f'lens_{d.j}'

            # We build up the comb list to get the covariance of it later
            # in the same order as our data points
            comb.append(d.object)

            theta = np.exp(d.object.meanlogr)
            npair = d.object.npairs
            weight = d.object.weight

            # account for double-counting
            if d.i == d.j:
                npair = npair/2
                weight = weight/2
            # xip / xim is a special case because it has two observables.
            # the other two are together below
            if d.corr_type == XI:
                xip = d.object.xip
                xim = d.object.xim
                xiperr = np.sqrt(d.object.varxip)
                ximerr = np.sqrt(d.object.varxim)
                n = len(xip)
                # add all the data points to the sacc
                for i in range(n):
                    S.add_data_point(XIP, (tracer1,tracer2), xip[i],
                        theta=theta[i], error=xiperr[i], npair=npair[i], weight= weight[i])
                    S.add_data_point(XIM, (tracer1,tracer2), xim[i],
                        theta=theta[i], error=ximerr[i], npair=npair[i], weight= weight[i])
            else:
                xi = d.object.xi
                err = np.sqrt(d.object.varxi)
                n = len(xi)
                for i in range(n):
                    S.add_data_point(d.corr_type, (tracer1,tracer2), xi[i],
                        theta=theta[i], error=err[i], weight=weight[i])

        # Add the covariance.  There are several different jackknife approaches
        # available - see the treecorr docs
        cov = treecorr.estimate_multi_cov(comb, self.config['var_method'])
        S.add_covariance(cov)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('twopoint_data_SCIA'), overwrite=True)

        if self.config['do_shear_pos'] == True:
            comb = []
            for d in results:
                tracer1 = f'source_{d.i}' 
                tracer2 = f'source_{d.j}'    
                
                if d.corr_type == GAMMAT:
                    theta = np.exp(d.object.meanlogr)
                    npair = d.object.npairs
                    weight = d.object.weight
                    xi_x = d.object.xi_im
                    covX = d.object.estimate_cov('shot')
                    comb.append(covX)
                    err = np.sqrt(np.diag(covX))
                    n = len(xi_x)
                    for i in range(n):
                        S2.add_data_point(GAMMAX, (tracer1,tracer2), xi_x[i],
                            theta=theta[i], error=err[i], weight=weight[i])
                if d.corr_type == GAMMATS:
                    theta = np.exp(d.object.meanlogr)
                    npair = d.object.npairs
                    weight = d.object.weight
                    xi_x = d.object.xi_im
                    covX = d.object.estimate_cov('shot')
                    comb.append(covX)
                    err = np.sqrt(np.diag(covX))
                    n = len(xi_x)
                    for i in range(n):
                        S2.add_data_point(GAMMAXS, (tracer1,tracer2), xi_x[i],
                            theta=theta[i], error=err[i], weight=weight[i])
            S2.add_covariance(comb)
            S2.to_canonical_order
            self.write_metadata(S2,meta)
            S2.save_fits(self.get_output('gammaX_scia'), overwrite=True)