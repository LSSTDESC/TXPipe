from ..twopoint import TXTwoPoint 
from ..data_types import HDFFile, ShearCatalog, TomographyCatalog, RandomsCatalog, FiducialCosmology, SACCFile, PhotozPDFFile, PNGFile, TextFile
from ..utils.calibration_tools import read_shear_catalog_type
import numpy as np
import random
import collections
import sys
from time import perf_counter

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
    the pairs where we have the shear object in front of the object used for density,
    these are the pairs we would expect should not contribute to the actual signal.  
    Once without imposing this selection funciton. 
    """
    name = 'TXSelfCalibrationIA'
    inputs = [
        ('calibrated_shear_catalog', ShearCatalog),
        ('binned_lens_catalog', HDFFile),
        ('binned_random_cats_source', HDFFile),
        ('shear_photoz_stack', HDFFile),
        ('lens_photoz_stack', HDFFile),
        ('patch_centers', TextFile),
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
        'bin_slop':0.0,
        'flip_g1': False,
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
        'low_mem': False,
        'metric': 'Rperp',
        }

    def run(self):

        super().run()

    def select_calculations(self, source_list, lens_list):
        calcs = []
        
        if self.config['do_shear_pos']:
            k = SHEAR_POS
            l = SHEAR_POS_SELECT # adding extra calls to do the selection function version for the shear_position. 
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

    def get_lens_catalog(self, i):
        raise ValueError(f"Something broke and TXPipe tried to load a lens catalog.")

    def get_shear_catalog(self, i):
        import treecorr
        import pyccl as ccl

        cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology'))
        cat_0 = treecorr.Catalog(
            self.get_input("calibrated_shear_catalog"),
            ext = f"/shear/bin_{i}",
            z_col = "z",
        )
        r = ccl.background.comoving_radial_distance(cosmo, 1/(1+cat_0.z))
        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            self.get_input("calibrated_shear_catalog"),
            ext = f"/shear/bin_{i}",
            g1_col = "g1",
            g2_col = "g2",
            r = r, 
            ra_col = "ra",
            dec_col = "dec",
            w_col = "weight",
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers'),
            flip_g1 = self.config["flip_g1"],
            flip_g2 = self.config["flip_g2"],
        )

        return cat

    def get_random_catalog(self, i):
        import treecorr
        import pyccl as ccl
        if not self.config["use_randoms"]:
            return None
        
        cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology'))
        cat_0 = treecorr.Catalog(
            self.get_input("calibrated_shear_catalog"),
            ext = f"/shear/bin_{i}",
            z_col = "z",
        )
        r = ccl.background.comoving_radial_distance(cosmo, 1/(1+cat_0.z))

        rancat = treecorr.Catalog(
            self.get_input("binned_random_cats"),
            ext = f"/randoms/bin_{i}",
            ra_col = "ra",
            dec_col = "dec",
            r = r,
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers'),
        ) 
        return rancat

    def calculate_shear_pos_select(self, i, j):
        # This is the added calculation that uses the selection function, as defined in our paper. it picks
        # out all the pairs where the object in the source catalog is in front of the lens object. 
        # Again the pairs picked out, are the pairs that should not be there. 
        # note we are looking at auto-correlations for the source bins!
        import treecorr 
        import pyccl as ccl

        cat_i = self.get_shear_catalog(i)
        n_i = cat_i.nobj

        cat_j = self.get_shear_catalog(j)
        rancat_j = self.get_random_catalog(j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        # NEW: we will calculate the separation in Mpc that corresponds to min_sep and max_sep, as if these were given in arcminutes!
        cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology')) # getting the cosmology
        r_mean_i = np.mean(cat_i.r) #getting the mean comoving distance in the bin
        a_i = ccl.scale_factor_of_chi(cosmo, r_mean_i) #getting the corresponding scale factor
        Da_i = ccl.angular_diameter_distance(cosmo, 1, a2= a_i) #calculating the angular diameter distance!
        config = self.config.copy() # copying the cofiguration options, so we don't overwrite the original configuration!
        config['min_sep'] = self.config['min_sep']*np.pi*Da_i /10_800
        config['max_sep'] = self.config['max_sep']*np.pi*Da_i /10_800


        if self.rank == 0:
            print(f"Calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")

        #Notice we are now calling config instead of self.config!
        ng = treecorr.NGCorrelation(config, max_rpar = 0.0)    # The max_rpar = 0.0, is in fact the same as our selection function. 
        t1 = perf_counter()
        ng.process(cat_j, cat_i, comm=self.comm)

        if rancat_j:
            rg = treecorr.NGCorrelation(config, max_rpar = 0.0)
            rg.process(rancat_j, cat_i, comm=self.comm)
        else:
            rg = None

        if self.rank == 0:
            ng.calculateXi(rg=rg)
            t2 = perf_counter()
            print(f"Processing took {t2 - t1:.1f} seconds")

        return ng

    def calculate_shear_pos(self, i, j):
        import treecorr
        import pyccl as ccl 

        cat_i = self.get_shear_catalog(i)
        n_i = cat_i.nobj

        cat_j = self.get_shear_catalog(j)
        rancat_j = self.get_random_catalog(j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        

        # NEW: we will calculate the separation in Mpc that corresponds to min_sep and max_sep, as if these were given in arcminutes!
        cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology')) # getting the cosmology
        r_mean_i = np.mean(cat_i.r) #getting the mean comoving distance in the bin
        a_i = ccl.scale_factor_of_chi(cosmo, r_mean_i) #getting the corresponding scale factor
        Da_i = ccl.angular_diameter_distance(cosmo, 1, a2= a_i) #calculating the angular diameter distance!
        config = self.config.copy() # copying the cofiguration options, so we don't overwrite the original configuration!
        config['min_sep'] = self.config['min_sep']*np.pi*Da_i /10_800
        config['max_sep'] = self.config['max_sep']*np.pi*Da_i /10_800

        if self.rank == 0:
            print(f"Calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")

        #Notice we are now calling config instead of self.config!
        ng = treecorr.NGCorrelation(config)
        t1 = perf_counter()
        ng.process(cat_j, cat_i, comm=self.comm)

        if rancat_j:
            rg = treecorr.NGCorrelation(config)
            rg.process(rancat_j, cat_i, comm = self.comm)
        else:
            rg = None

        if self.rank == 0:
            ng.calculateXi(rg=rg)
            t2 = perf_counter()
            print(f"Processing took {t2 - t1:.1f} seconds")

        ng.calculateXi(rg=rg)

        return ng

    def calculate_pos_pos(self, i, j):
        import pyccl as ccl
        import treecorr

        cat_i, = self.get_shear_catalog(i)
        rancat_i = self.get_random_catalog(i)
        n_i = cat_i.nobj
        n_rand_i = rancat_i.nobj if rancat_i is not None else 0

        if i==j:
            cat_j = None
            rancat_j = rancat_i

        # NEW: we will calculate the separation in Mpc that corresponds to min_sep and max_sep, as if these were given in arcminutes!
        cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology')) # getting the cosmology
        r_mean_i = np.mean(cat_i.r) #getting the mean comoving distance in the bin
        a_i = ccl.scale_factor_of_chi(cosmo, r_mean_i) #getting the corresponding scale factor
        Da_i = ccl.angular_diameter_distance(cosmo, 1, a2= a_i) #calculating the angular diameter distance!
        config = self.config.copy() # copying the cofiguration options, so we don't overwrite the original configuration!
        config['min_sep'] = self.config['min_sep']*np.pi*Da_i /10_800
        config['max_sep'] = self.config['max_sep']*np.pi*Da_i /10_800

        if self.rank == 0:
            print(f"Calculating position-position bin pair ({i}, {j}): {n_i} x {n_i} objects,  {n_rand_i} randoms")


        nn = treecorr.NNCorrelation(config)
        nn.process(cat_i, cat_j, comm=self.comm)
        
        nr = treecorr.NNCorrelation(config)
        nr.process(cat_i, rancat_j, comm=self.comm)

        # The next calculation is faster if we explicitly tell TreeCorr
        # that its two catalogs here are the same one.
        if i == j:
            rancat_j = None
        
        rr = treecorr.NNCorrelation(config)
        rr.process(rancat_i, rancat_j, comm=self.comm)
        
        if i==j:
            rn = None
        else:
            rn = treecorr.NNCorrelation(config)
            rn.process(rancat_i, cat_j, comm=self.comm)


        if self.rank == 0:
            t2 = perf_counter()
            nn.calculateXi(rr, dr=nr, rd=rn)
            print(f"Processing took {t2 - t1:.1f} seconds")   

        return nn


    def call_treecorr(self, i, j, k):
        import sacc 

        if k==SHEAR_SHEAR:
            xx = self.calculate_shear_shear(i, j)
            xtype = "combined"
        elif k==SHEAR_POS:
            xx = self.calculate_shear_pos(i, j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k==SHEAR_POS_SELECT: #added the call to the selection function calculation.
            xx = self.calculate_shear_pos_select( i, j)
            xtype = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='ts')
        elif k==POS_POS:
            xx = self.calculate_pos_pos( i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        else:
            raise ValueError(f"Unknown correlation function {k}")

        result = Measurement(xtype, xx, i, j)

        sys.stdout.flush()
        return result

    def write_output(self, source_list, lens_list, meta, results):
        import sacc
        import treecorr
        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        # We define a new sacc data type for our selection function results.
        GAMMATS = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='ts')
        WTHETA = sacc.standard_types.galaxy_density_xi
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        # We must add these new data types for both the ts result and the xs. 
        GAMMAXS = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='xs')

        S = sacc.Sacc()
        if self.config['do_shear_pos'] == True:
            S2 = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data
        f = self.open_input('shear_photoz_stack')

        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        for i in source_list:
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            S.add_tracer('NZ', f'source_{i}', z, Nz)
            if self.config['do_shear_pos'] == True:
                S2.add_tracer('NZ', f'source_{i}', z, Nz)


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
                        theta=theta[i], error=err[i], npair=npair[i], weight=weight[i])

        # Add the covariance.  There are several different jackknife approaches
        # available - see the treecorr docs
        cov = treecorr.estimate_multi_cov(comb, self.config['var_method'])
        S.add_covariance(cov)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()
        self.write_metadata(S,meta)
        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('twopoint_data_SCIA'), overwrite=True)

        # In the case we do shear_position we can also look at the gamma_x product,
        # We expect this to be a null test, but it should still be saved. To not mess with
        # how the covariance is structured we save these in a seperate file here.  
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