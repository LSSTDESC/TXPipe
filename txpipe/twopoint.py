from .base_stage import PipelineStage
from .data_types import HDFFile, ShearCatalog, TomographyCatalog, RandomsCatalog, FiducialCosmology, SACCFile, PhotozPDFFile, PNGFile, TextFile
from .utils.calibration_tools import apply_metacal_response, apply_lensfit_calibration 
from .utils.calibration_tools import read_shear_catalog_type
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



class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('calibrated_shear_catalog', ShearCatalog),
        ('calibrated_lens_catalog', HDFFile),
        ('binned_random_cats', HDFFile),
        ('shear_photoz_stack', HDFFile),
        ('lens_photoz_stack', HDFFile),
        ('patch_centers', TextFile),
        ('tracer_metadata', HDFFile),
    ]
    outputs = [
        ('twopoint_data_real_raw', SACCFile),
        ('twopoint_gamma_x', SACCFile)
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        # TODO: Allow more fine-grained selection of 2pt subsets to compute
        'calcs':[0,1,2],
        'min_sep':0.5,
        'max_sep':300.,
        'nbins':9,
        'bin_slop':0.0,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'source_bins':[-1],
        'lens_bins':[-1],
        'reduce_randoms_size':1.0,
        'do_shear_shear': True,
        'do_shear_pos': True,
        'do_pos_pos': True,
        'var_method': 'jackknife',
        'use_true_shear': False,
        'subtract_mean_shear':False,
        'use_randoms': True,
        'low_mem': False,
        }

    def run(self):
        """
        Run the analysis for this stage.
        """
        import sacc
        import healpy
        import treecorr
        # Binning information
        source_list, lens_list = self.read_nbin()

        # Calculate metadata like the area and related
        # quantities
        meta = self.read_metadata()

        # Choose which pairs of bins to calculate
        calcs = self.select_calculations(source_list, lens_list)
        sys.stdout.flush()

        # This splits the calculations among the parallel bins
        # It's not necessarily the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment,
        # but for this case I would expect it to be mostly fine
        results = []
        for i,j,k in calcs:
            result = self.call_treecorr(i, j, k)
            results.append(result)

        # If we are running in parallel this collects the results together
        #results = self.collect_results(results)

        # Save the results
        if self.rank==0:
            self.write_output(source_list, lens_list, meta, results)


    def select_calculations(self, source_list, lens_list):
        calcs = []

        # For shear-shear we omit pairs with j>i
        if self.config['do_shear_shear']:
            k = SHEAR_SHEAR
            for i in source_list:
                for j in range(i+1):
                    if j in source_list:
                        calcs.append((i,j,k))

        # For shear-position we use all pairs
        if self.config['do_shear_pos']:
            k = SHEAR_POS
            for i in source_list:
                for j in lens_list:
                    calcs.append((i,j,k))

        # For position-position we omit pairs with j>i
        if self.config['do_pos_pos']:
            if not self.config['use_randoms']:
                raise ValueError("You need to have a random catalog to calculate position-position correlations")
            k = POS_POS
            for i in lens_list:
                for j in range(i+1):
                    if j in lens_list:
                        calcs.append((i,j,k))

        if self.rank==0:
            print(f"Running these calculations: {calcs}")

        return calcs

    def collect_results(self, results):
        if self.comm is None:
            return results

        results = self.comm.gather(results, root=0)

        # Concatenate results on master
        if self.rank==0:
            results = sum(results, [])

        return results

    def read_nbin(self):
        """
        Determine the bins to use in this analysis, either from the input file
        or from the configuration.
        """
        if self.config['source_bins'] == [-1] and self.config['lens_bins'] == [-1]:
            source_list, lens_list = self._read_nbin_from_tomography()
        else:
            source_list, lens_list = self._read_nbin_from_config()

        ns = len(source_list)
        nl = len(lens_list)
        print(f'Running with {ns} source bins and {nl} lens bins')

        return source_list, lens_list


    # These two functions can be combined into a single one.
    def _read_nbin_from_tomography(self):
        with self.open_input('calibrated_shear_catalog') as f:
            nbin_source = f['shear'].attrs['nbin_source']

        with self.open_input('calibrated_lens_catalog') as f:
            nbin_lens = f['lens'].attrs['nbin_lens']

        source_list = range(nbin_source)
        lens_list = range(nbin_lens)

        return source_list, lens_list

    def _read_nbin_from_config(self):
        # TODO handle the case where the user only specefies
        # bins for only sources or only lenses
        source_list = self.config['source_bins']
        lens_list = self.config['lens_bins']

        # catch bad input
        tomo_source_list, tomo_lens_list = self._read_nbin_from_tomography()
        tomo_nbin_source = len(tomo_source_list)
        tomo_nbin_lens = len(tomo_lens_list)

        nbin_source = len(source_list)
        nbin_lens = len(lens_list) 

        if source_list == [-1]:
            source_list = tomo_source_list
        if lens_list == [-1]:
            lens_list = tomo_lens_list

        # if more bins are input than exist, raise an error
        if not nbin_source <= tomo_nbin_source:
            raise ValueError(f'Requested too many source bins in the config ({nbin_source}): max is {tomo_nbin_source}')
        if not nbin_lens <= tomo_nbin_lens:
            raise ValueError(f'Requested too many lens bins in the config ({nbin_lens}): max is {tomo_nbin_lens}')

        # make sure the bin numbers actually exist
        for i in source_list:
            if i not in tomo_source_list:
                raise ValueError(f"Requested source bin {i} that is not in the input file")

        for i in lens_list:
            if i not in tomo_lens_list:
                raise ValueError(f"Requested lens bin {i} that is not in the input file")

        return source_list, lens_list



    def write_output(self, source_list, lens_list, meta, results):
        import sacc
        import treecorr
        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMAX = sacc.standard_types.galaxy_shearDensity_xi_x
        WTHETA = sacc.standard_types.galaxy_density_xi

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
                S2.add_tracer('NZ', f'source_{i}',z, Nz)

        f = self.open_input('lens_photoz_stack')
        # For both source and lens
        for i in lens_list:
            z = f['n_of_z/lens/z'][:]
            Nz = f[f'n_of_z/lens/bin_{i}'][:]
            S.add_tracer('NZ', f'lens_{i}', z, Nz)
            if self.config['do_shear_pos'] == True:
                S2.add_tracer('NZ', f'lens_{i}',z, Nz)
        # Closing n(z) file
        f.close()

        # Now build up the collection of data points, adding them all to
        # the sacc data one by one.
        comb = []
        for d in results:
            # First the tracers and generic tags
            tracer1 = f'source_{d.i}' if d.corr_type in [XI, GAMMAT] else f'lens_{d.i}'
            tracer2 = f'source_{d.j}' if d.corr_type in [XI] else f'lens_{d.j}'

            # We build up the comb list to get the covariance of it later
            # in the same order as our data points
            comb.append(d.object)

            theta = np.exp(d.object.meanlogr)
            npair = d.object.npairs
            weight = d.object.weight
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
                for i in range(n):                    
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

        self.write_metadata(S,meta)

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('twopoint_data_real_raw'), overwrite=True)
        
        # Adding the gammaX calculation:
        
        if self.config['do_shear_pos'] == True:
            comb = []
            for d in results:
                tracer1 = f'source_{d.i}' if d.corr_type in [XI, GAMMAT] else f'lens_{d.i}'
                tracer2 = f'source_{d.j}' if d.corr_type in [XI] else f'lens_{d.j}'   
                
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
            S2.add_covariance(comb)
            S2.to_canonical_order
            self.write_metadata(S2,meta)
            S2.save_fits(self.get_output('twopoint_gamma_x'), overwrite=True)



    def write_metadata(self, S, meta):
        # We also save the associated metadata to the file
        for k,v in meta.items():
            if np.isscalar(v):
                S.metadata[k] = v
            else:
                for i, vi in enumerate(v):
                    S.metadata[f'{k}_{i}'] = vi

        # Add provenance metadata.  In managed formats this is done
        # automatically, but because the Sacc library is external
        # we do it manually here.
        provenance = self.gather_provenance()
        provenance.update(SACCFile.generate_provenance())
        for key, value in provenance.items():
            if isinstance(value, str) and '\n' in value:
                values = value.split("\n")
                for i,v in enumerate(values):
                    S.metadata[f'provenance/{key}_{i}'] = v
            else:
                S.metadata[f'provenance/{key}'] = value


    def call_treecorr(self, i, j, k):
        """
        This is a wrapper for interaction with treecorr.
        """
        import sacc


        if k==SHEAR_SHEAR:
            xx = self.calculate_shear_shear(i, j)
            xtype = "combined"
        elif k==SHEAR_POS:
            xx = self.calculate_shear_pos(i, j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k==POS_POS:
            xx = self.calculate_pos_pos(i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        else:
            raise ValueError(f"Unknown correlation function {k}")

        result = Measurement(xtype, xx, i, j)

        sys.stdout.flush()
        return result


    def get_shear_catalog(self, i):
        import treecorr

        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            self.get_input("calibrated_shear_catalog"),
            ext = f"/shear/bin_{i}",
            g1_col = "g1",
            g2_col = "g2",
            ra_col = "ra",
            dec_col = "dec",
            w_col = "weight",
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers')
        )
        return cat


    def get_lens_catalog(self, i):
        import treecorr

        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            self.get_input("calibrated_lens_catalog"),
            ext = f"/lens/bin_{i}",
            ra_col = "ra",
            dec_col = "dec",
            w_col = "weight",
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers')
        )
        return cat

    def get_random_catalog(self, i):
        import treecorr
        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("binned_random_cats"),
            ext = f"/randoms/bin_{i}",
            ra_col = "ra",
            dec_col = "dec",
            w_col = "weight",
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers')
        ) 
        return rancat


    def calculate_shear_shear(self, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        n_i = cat_i.nobj

        if i==j:
            cat_j = None
            n_j = n_i
        else:
            cat_j = self.get_shear_catalog(j)
            n_j = cat_j.nobj

        print(f"Rank {self.rank} calculating shear-shear bin pair ({i},{j}): {n_i} x {n_j} objects")

        gg = treecorr.GGCorrelation(self.config)
        t1 = perf_counter()
        gg.process(cat_i, cat_j, low_mem=self.config["low_mem"], comm=self.comm)
        t2 = perf_counter()
        if self.rank == 0:
            print(f"Processing took {t2 - t1:.1f} seconds")

        return gg

    def calculate_shear_pos(self, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        n_i = cat_i.nobj

        cat_j = self.get_lens_catalog(j)
        rancat_j = self.get_random_catalog(j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        print(f"Rank {self.rank} calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")

        ng = treecorr.NGCorrelation(self.config)
        ng.process(cat_j, cat_i, comm=self.comm)

        if rancat_j:
            rg = treecorr.NGCorrelation(self.config)
            rg.process(rancat_j, cat_i, comm=self.comm)
        else:
            rg = None

        if self.rank == 0:
            ng.calculateXi(rg=rg)

        return ng


    def calculate_pos_pos(self, i, j):
        import treecorr

        cat_i = self.get_lens_catalog(i)
        rancat_i = self.get_random_catalog(i)
        n_i = cat_i.nobj
        n_rand_i = rancat_i.nobj if rancat_i is not None else 0

        nn = treecorr.NNCorrelation(self.config)
        rn = treecorr.NNCorrelation(self.config)
        nr = treecorr.NNCorrelation(self.config)
        rr = treecorr.NNCorrelation(self.config)
        
        if i==j:
            cat_j = None
            rancat_j = None
        else:
            cat_j = self.get_lens_catalog(j)
            rancat_j = self.get_random_catalog(j)

        nn = treecorr.NNCorrelation(self.config)
        nn.process(cat_i, cat_j, comm=self.comm)

        nr = treecorr.NNCorrelation(self.config)
        nr.process(cat_i, rancat_j, comm=self.comm)

        rr = treecorr.NNCorrelation(self.config)
        rr.process(rancat_i, rancat_j, comm=self.comm)

        if i==j:
            rn = None
        else:
            rn = treecorr.NNCorrelation(self.config)
            rn.process(rancat_i, cat_j, comm=self.comm)

        if self.rank == 0:
            nn.calculateXi(rr, dr=nr, rd=rn)
        return nn




    def calculate_area(self, data):
        import healpy as hp
        pix=hp.ang2pix(4096, np.pi/2.-np.radians(data['dec']),np.radians(data['ra']), nest=True)
        area=hp.nside2pixarea(4096)*(180./np.pi)**2
        mask=np.bincount(pix)>0
        area=np.sum(mask)*area
        area=float(area) * 60. * 60.
        return area

    def read_metadata(self):
        meta_data = self.open_input('tracer_metadata')
        area = meta_data['tracers'].attrs['area']
        sigma_e = meta_data['tracers/sigma_e'][:]
        N_eff = meta_data['tracers/N_eff'][:]
        mean_e1 = meta_data['tracers/mean_e1'][:]
        mean_e2 = meta_data['tracers/mean_e2'][:]

        meta = {}
        meta["neff"] =  N_eff
        meta["area"] =  area
        meta["sigma_e"] =  sigma_e
        meta["mean_e1"] = mean_e1
        meta["mean_e2"] = mean_e2

        return meta

    

    
class TXTwoPointPlots(PipelineStage):
    """
    Make n(z) plots
    """
    name='TXTwoPointPlots'
    inputs = [
        ('twopoint_data_real', SACCFile),
        ('fiducial_cosmology', FiducialCosmology),  # For example lines
        ('twopoint_gamma_x', SACCFile),
        ('twopoint_theory_real', SACCFile),
    ]
    outputs = [
        ('shear_xi_plus', PNGFile),
        ('shear_xi_minus', PNGFile),
        ('shearDensity_xi', PNGFile),
        ('density_xi', PNGFile),
        ('shear_xi_plus_ratio', PNGFile),
        ('shear_xi_minus_ratio', PNGFile),
        ('shearDensity_xi_ratio', PNGFile),
        ('density_xi_ratio', PNGFile),
        ('shearDensity_xi_x', PNGFile),
    ]

    config_options = {
        'wspace': 0.05,
        'hspace': 0.05,
    }


    def run(self):
        import sacc
        import matplotlib
        import pyccl
        from .plotting import full_3x2pt_plots
        matplotlib.use('agg')
        matplotlib.rcParams["xtick.direction"]='in'
        matplotlib.rcParams["ytick.direction"]='in'

        filename = self.get_input('twopoint_data_real')
        s = sacc.Sacc.load_fits(filename)
        nbin_source, nbin_lens = self.read_nbin(s)

        filename_theory = self.get_input('twopoint_theory_real')

        outputs = {
            "galaxy_density_xi": self.open_output('density_xi',
                figsize=(3.5*nbin_lens, 3*nbin_lens), wrapper=True),

            "galaxy_shearDensity_xi_t": self.open_output('shearDensity_xi',
                figsize=(3.5*nbin_lens, 3*nbin_source), wrapper=True),

            "galaxy_shear_xi_plus": self.open_output('shear_xi_plus',
                figsize=(3.5*nbin_source, 3*nbin_source), wrapper=True),

            "galaxy_shear_xi_minus": self.open_output('shear_xi_minus',
                figsize=(3.5*nbin_source, 3*nbin_source), wrapper=True),
            
        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots([filename], ['twopoint_data_real'], figures=figures, 
                         theory_sacc_files=[filename_theory], theory_labels=['Fiducial'])

        for fig in outputs.values():
            fig.close()

        outputs = {
            "galaxy_density_xi": self.open_output('density_xi_ratio',
                figsize=(3.5*nbin_lens, 3*nbin_lens), wrapper=True),

            "galaxy_shearDensity_xi_t": self.open_output('shearDensity_xi_ratio',
                figsize=(3.5*nbin_lens, 3*nbin_source), wrapper=True),

            "galaxy_shear_xi_plus": self.open_output('shear_xi_plus_ratio',
                figsize=(3.5*nbin_source, 3*nbin_source), wrapper=True),

            "galaxy_shear_xi_minus": self.open_output('shear_xi_minus_ratio',
                figsize=(3.5*nbin_source, 3*nbin_source), wrapper=True),
            
        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots([filename], ['twopoint_data_real'], figures=figures,
                         theory_sacc_files=[filename_theory], theory_labels=['Fiducial'], ratios=True)

        for fig in outputs.values():
            fig.close()

            
        filename = self.get_input('twopoint_gamma_x')

        outputs = {
            "galaxy_shearDensity_xi_x": self.open_output('shearDensity_xi_x',
                figsize=(3.5*nbin_lens, 3*nbin_source), wrapper=True),
        }
        
        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots([filename], ['twopoint_gamma_x'], 
            figures=figures)

        for fig in outputs.values():
            fig.close()

    def read_nbin(self, s):
        import sacc

        xip = sacc.standard_types.galaxy_shear_xi_plus
        wtheta = sacc.standard_types.galaxy_density_xi

        source_tracers = set()
        for b1, b2 in s.get_tracer_combinations(xip):
            source_tracers.add(b1)
            source_tracers.add(b2)

        lens_tracers = set()
        for b1, b2 in s.get_tracer_combinations(wtheta):
            lens_tracers.add(b1)
            lens_tracers.add(b2)


        return len(source_tracers), len(lens_tracers)

    def read_bins(self, s):
        import sacc

        xip = sacc.standard_types.galaxy_shear_xi_plus
        wtheta = sacc.standard_types.galaxy_density_xi

        source_tracers = set()
        for b1, b2 in s.get_tracer_combinations(xip):
            source_tracers.add(b1)
            source_tracers.add(b2)

        lens_tracers = set()
        for b1, b2 in s.get_tracer_combinations(wtheta):
            lens_tracers.add(b1)
            lens_tracers.add(b2)

        sources = list(sorted(source_tracers))
        lenses = list(sorted(lens_tracers))

        return sources, lenses

    
    def get_theta_xi_err(self, D):
        """
        For a given datapoint D, returns theta, xi, err,
        after masking for positive errorbars
        (sometimes there are NaNs).
        """
        theta = np.array([d.get_tag('theta') for d in D])
        xi    = np.array([d.value for d in D])
        err   = np.array([d.get_tag('error') for d  in D])
        w = err>0
        theta = theta[w]
        xi = xi[w]
        err = err[w]

        return theta, xi, err


    def get_theta_xi_err_jk(self, s, dt, src1, src2):
        """
        In this case we want to get the JK errorbars,
        which are stored in the covariance, so we want to
        load a particular covariance block, given a dataype dt.
        Returns theta, xi, err,
        after masking for positive errorbars
        (sometimes there are NaNs).
        """
        theta_jk, xi_jk, cov_jk = s.get_theta_xi(dt, src1, src2, return_cov = True)
        err_jk = np.sqrt(np.diag(cov_jk))
        w_jk = err_jk>0
        theta_jk = theta_jk[w_jk]
        xi_jk = xi_jk[w_jk]
        err_jk = err_jk[w_jk]
        
        return theta_jk, xi_jk, err_jk


    
    def plot_shear_shear(self, s, sources):
        import sacc
        import matplotlib.pyplot as plt

        xip = sacc.standard_types.galaxy_shear_xi_plus
        xim = sacc.standard_types.galaxy_shear_xi_minus
        nsource = len(sources)


        theta = s.get_tag('theta', xip)
        tmin = np.min(theta)
        tmax = np.max(theta)

        coord = lambda dt,i,j: (nsource+1-j, i) if dt==xim else (j, nsource-1-i)

        for dt in [xip, xim]:
            for i,src1 in enumerate(sources[:]):
                for j,src2 in enumerate(sources[:]):
                    D = s.get_data_points(dt, (src1,src2))


                    if len(D)==0:
                        continue

                    ax = plt.subplot2grid((nsource+2, nsource), coord(dt,i,j))

                    scale = 1e-4

                    theta = np.array([d.get_tag('theta') for d in D])
                    xi    = np.array([d.value for d in D])
                    err   = np.array([d.get_tag('error') for d  in D])
                    w = err>0
                    theta = theta[w]
                    xi = xi[w]
                    err = err[w]

                    plt.errorbar(theta, xi*theta / scale, err*theta / scale, fmt='.')
                    plt.xscale('log')
                    plt.ylim(-1,1)
                    plt.xlim(tmin, tmax)

                    if dt==xim:
                        if j>0:
                            ax.set_xticklabels([])
        plots = ['xi', 'xi_err']

        for plot in plots:
            plot_output = self.open_output(f'shear_{plot}', wrapper=True, figsize=(2.5*nsource,2*nsource))

            for dt in [xip, xim]:
                for i,src1 in enumerate(sources[:]):
                    for j,src2 in enumerate(sources[:]):
                        D = s.get_data_points(dt, (src1,src2))

                        if len(D)==0:
                            continue

                        ax = plt.subplot2grid((nsource+2, nsource), coord(dt,i,j))

                        theta, xi, err = self.get_theta_xi_err(D)
                        if plot == 'xi':
                            scale = 1e-4
                            plt.errorbar(theta, xi*theta / scale, err*theta / scale, fmt='.',
                                         capsize=1.5,color = self.colors[0])
                            plt.ylim(-30,30)
                            ylabel_xim = r'$\theta \cdot \xi_{-} \cdot 10^4$'
                            ylabel_xip = r'$\theta \cdot \xi_{+} \cdot 10^4$'

                        if plot == 'xi_err':
                            theta_jk, xi_jk, err_jk = self.get_theta_xi_err_jk(s, dt, src1, src2)
                            plt.plot(theta, err, label = 'Shape noise', lw = 2., color = self.colors[0])
                            plt.plot(theta_jk, err_jk, label = 'Jackknife', lw = 2., color = self.colors[1])
                            ylabel_xim = r'$\sigma\, (\xi_{-})$'
                            ylabel_xip = r'$\sigma\, (\xi_{-})$'
                            
                        plt.xscale('log')
                        plt.xlim(tmin, tmax)

                        if dt==xim:
                            if j>0:
                                ax.set_xticklabels([])
                            else:
                                plt.xlabel(r'$\theta$ (arcmin)')

                            if i==nsource-1:
                                ax.yaxis.tick_right()
                                ax.yaxis.set_label_position("right")
                                ax.set_ylabel(ylabel_xim)
                            else:
                                ax.set_yticklabels([])
                        else:
                            ax.set_xticklabels([])
                            if i==nsource-1:
                                ax.set_ylabel(ylabel_xip)
                            else:
                                ax.set_yticklabels([])

                        #props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                        plt.text(0.03, 0.93, f'[{i},{j}]', transform=plt.gca().transAxes,
                            fontsize=10, verticalalignment='top')#, bbox=props)

            if plot == 'xi_err':
                plt.legend()
            plt.tight_layout()
            plt.subplots_adjust(hspace=self.config['hspace'],wspace=self.config['wspace'])
            plot_output.close()

            
    def plot_shear_density(self, s, sources, lenses):
        import sacc
        import matplotlib.pyplot as plt

        gammat = sacc.standard_types.galaxy_shearDensity_xi_t
        nsource = len(sources)
        nlens = len(lenses)

        theta = s.get_tag('theta', gammat)
        tmin = np.min(theta)
        tmax = np.max(theta)

        plots = ['xi', 'xi_err']
        for plot in plots:
            plot_output = self.open_output(f'shearDensity_{plot}', wrapper=True, figsize=(3*nlens,2*nsource))

            for i,src1 in enumerate(sources):
                for j,src2 in enumerate(lenses):
                    
                    D = s.get_data_points(gammat, (src1,src2))

                    if len(D)==0:
                        continue
                    
                    ax = plt.subplot2grid((nsource, nlens), (i,j))

                    if plot == 'xi':
                        scale = 1e-2
                        theta, xi, err = self.get_theta_xi_err(D)
                        plt.errorbar(theta, xi*theta / scale, err*theta / scale, fmt='.',
                                     capsize=1.5, color = self.colors[0])
                        plt.ylim(-2,2)
                        ylabel = r"$\theta \cdot \gamma_t \cdot 10^2$"
                            
                    if plot == 'xi_err':
                        theta, xi, err = self.get_theta_xi_err(D)
                        theta_jk, xi_jk, err_jk = self.get_theta_xi_err_jk(s, gammat, src1, src2)
                        plt.plot(theta, err, label = 'Shape noise', lw =2., color = self.colors[0])
                        plt.plot(theta_jk, err_jk, label = 'Jackknife', lw =2., color = self.colors[1])
                        ylabel = r"$\sigma\,(\gamma_t)$"
                    
                    plt.xscale('log')
                    plt.xlim(tmin, tmax)

                    if i==nsource-1:
                        plt.xlabel(r'$\theta$ (arcmin)')
                    else:
                        ax.set_xticklabels([])

                    if j==0:
                        plt.ylabel(ylabel)
                    else:
                        ax.set_yticklabels([])

                    #props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                    plt.text(0.03, 0.93, f'[{i},{j}]', transform=plt.gca().transAxes,
                             fontsize=10, verticalalignment='top')#, bbox=props)

            if plot == 'xi_err':
                plt.legend()
            plt.tight_layout()
            plt.subplots_adjust(hspace=self.config['hspace'],wspace=self.config['wspace'])
            plot_output.close()



    def plot_density_density(self, s, lenses):
        import sacc
        import matplotlib.pyplot as plt

        wtheta = sacc.standard_types.galaxy_density_xi
        nlens = len(lenses)

        theta = s.get_tag('theta', wtheta)
        tmin = np.min(theta)
        tmax = np.max(theta)

        plots = ['xi', 'xi_err']
        for plot in plots:
            plot_output = self.open_output(f'density_{plot}', wrapper=True, figsize=(3*nlens,2*nlens))
         
            for i,src1 in enumerate(lenses[:]):
                for j,src2 in enumerate(lenses[:]):

                    D = s.get_data_points(wtheta, (src1,src2))

                    if len(D)==0:
                        continue

                    ax = plt.subplot2grid((nlens, nlens), (i,j))

                    if plot == 'xi':
                        scale = 1
                        theta, xi, err = self.get_theta_xi_err(D)
                        plt.errorbar(theta, xi*theta / scale, err*theta / scale, fmt='.',
                                     capsize=1.5, color = self.colors[0])
                        ylabel = r"$\theta \cdot w$"
                        plt.ylim(-1,1)
                            
                    if plot == 'xi_err':
                        theta, xi, err = self.get_theta_xi_err(D)
                        theta, xi, err = self.get_theta_xi_err(D)
                        theta_jk, xi_jk, err_jk = self.get_theta_xi_err_jk(s, wtheta, src1, src2)
                        plt.plot(theta, err, label = 'Shape noise', lw =2., color = self.colors[0])
                        plt.plot(theta_jk, err_jk, label = 'Jackknife', lw =2., color = self.colors[1])
                        ylabel = r"$\sigma\,(w)$"

                    plt.xscale('log')
                    plt.xlim(tmin, tmax)

                    if j>0:
                        ax.set_xticklabels([])
                    else:
                        plt.xlabel(r'$\theta$ (arcmin)')

                    if i==0:
                        plt.ylabel(ylabel)
                    else:
                        ax.set_yticklabels([])

                    #props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                    plt.text(0.03, 0.93, f'[{i},{j}]', transform=plt.gca().transAxes,
                        fontsize=10, verticalalignment='top')#, bbox=props)

            if plot == 'xi_err':
                plt.legend()
            plt.tight_layout()
            plt.subplots_adjust(hspace=self.config['hspace'],wspace=self.config['wspace'])
            plot_output.close()



if __name__ == '__main__':
    PipelineStage.main()
