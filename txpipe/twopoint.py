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
        ('binned_shear_catalog', ShearCatalog),
        ('binned_lens_catalog', HDFFile),
        ('binned_random_catalog', HDFFile),
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
        'flip_g1':False,
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


        # Compute all the requested correlation functions,
        # using MPI if available
        results = []
        cache = {}
        for i,j,k in calcs:
            result = self.call_treecorr(i, j, k, cache)
            results.append(result)


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
            print(f"Running {len(calcs)} calculations: {calcs}")

        return calcs


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
        if self.rank == 0:
            print(f'Running with {ns} source bins and {nl} lens bins')

        return source_list, lens_list


    # These two functions can be combined into a single one.
    def _read_nbin_from_tomography(self):
        with self.open_input('binned_shear_catalog') as f:
            nbin_source = f['shear'].attrs['nbin_source']

        with self.open_input('binned_lens_catalog') as f:
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


    def call_treecorr(self, i, j, k, cache):
        """
        This is a wrapper for interaction with treecorr.
        """
        import sacc


        if k==SHEAR_SHEAR:
            xx = self.calculate_shear_shear(i, j, cache)
            xtype = "combined"
        elif k==SHEAR_POS:
            xx = self.calculate_shear_pos(i, j, cache)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k==POS_POS:
            xx = self.calculate_pos_pos(i, j, cache)
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
            self.get_input("binned_shear_catalog"),
            ext = f"/shear/bin_{i}",
            g1_col = "g1",
            g2_col = "g2",
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


    def get_lens_catalog(self, i):
        import treecorr

        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            self.get_input("binned_lens_catalog"),
            ext = f"/lens/bin_{i}",
            ra_col = "ra",
            dec_col = "dec",
            w_col = "weight",
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers'),
        )
        return cat

    def get_random_catalog(self, i):
        import treecorr
        if not self.config["use_randoms"]:
            return None

        rancat = treecorr.Catalog(
            self.get_input("binned_random_catalog"),
            ext = f"/randoms/bin_{i}",
            ra_col = "ra",
            dec_col = "dec",
            ra_units='degree',
            dec_units='degree',
            patch_centers=self.get_input('patch_centers'),
        ) 
        return rancat

    @staticmethod
    def get_catalog_tag(kind, cat1, cat2):
        if cat1.name == '' or ((cat2 is not None) and (cat2.name == '')):
            return None
        if 'ext' not in cat1.config or ((cat2 is not None) and ('ext' not in cat2.config)):
            return None

        if cat2 is None:
            return f'{kind}_auto_{cat1.name}_{cat1.config["ext"]}'
        else:
            return f'{kind}_{cat1.name}_{cat1.config["ext"]}--{cat2.name}_{cat2.config["ext"]}'

    def process(self, kind, cat_i, cat_j, cache=None):

        key = self.get_catalog_tag(kind, cat_i, cat_j)
        if key and (cache is not None):
            return cache[key]

        if kind == 'nn':
            corr = treecorr.NNCorrelation(self.config)
        elif kind == 'ng':
            corr = treecorr.NGCorrelation(self.config)
        elif kind == 'gg':
            corr = treecorr.GGCorrelation(self.config)
        else:
            raise ValueError(f"Unknown correlation kind {kind}")

        corr.process(cat_i, cat_j, comm=self.comm, low_mem=self.config["low_mem"])

        if key and (cache is not None):
            cache[key] = corr

        return corr


    def calculate_shear_shear(self, i, j, cache=None):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        n_i = cat_i.nobj

        if i==j:
            cat_j = None
            n_j = n_i
        else:
            cat_j = self.get_shear_catalog(j)
            n_j = cat_j.nobj

        if self.rank == 0:
            print(f"Calculating shear-shear bin pair ({i},{j}): {n_i} x {n_j} objects using MPI")

        t1 = perf_counter()
        gg = self.process('gg', cat_i, cat_j, cache)
        t2 = perf_counter()
        if self.rank == 0:
            print(f"Processing took {t2 - t1:.1f} seconds")

        return gg

    def calculate_shear_pos(self, i, j, cache=None):
        import treecorr

        cat_i = self.get_shear_catalog(i)
        n_i = cat_i.nobj

        cat_j = self.get_lens_catalog(j)
        rancat_j = self.get_random_catalog(j)
        n_j = cat_j.nobj
        n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        if self.rank == 0:
            print(f"Calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand_j} randoms")


        t1 = perf_counter()
        ng = self.process('ng', cat_j, cat_i, cache)

        if rancat_j:
            rg = self.process('ng', rancat_j, cat_i, cache)
        else:
            rg = None


        if self.rank == 0:
            ng.calculateXi(rg=rg)
            t2 = perf_counter()
            print(f"Processing took {t2 - t1:.1f} seconds")

        
        return ng



    def calculate_pos_pos(self, i, j, cache=None):
        import treecorr

        cat_i = self.get_lens_catalog(i)
        rancat_i = self.get_random_catalog(i)
        n_i = cat_i.nobj
        n_rand_i = rancat_i.nobj if rancat_i is not None else 0
        
        if i==j:
            cat_j = None
            rancat_j = rancat_i
            n_j = n_i
            n_rand_j = n_rand_i
        else:
            cat_j = self.get_lens_catalog(j)
            rancat_j = self.get_random_catalog(j)
            n_j = cat_j.nobj
            n_rand_j = rancat_j.nobj


        if self.rank == 0:
            print(f"Calculating position-position bin pair ({i}, {j}): {n_i} x {n_j} objects,  {n_rand_i} x {n_rand_j} randoms")

        t1 = perf_counter()
        
        nn = self.process('nn', cat_i, cat_j, cache)
        nr = self.process('nn', cat_i, rancat_j, cache)
        
        # The next calculation is faster if we explicitly tell TreeCorr
        # that its two catalogs here are the same one.
        if i == j:
            rancat_j = None
        
        rr = self.process('nn', rancat_i, rancat_j, cache)
        
        if i==j:
            rn = None
        else:
            rn = self.process('nn', rancat_i, cat_j, cache)

        if self.rank == 0:
            t2 = perf_counter()
            nn.calculateXi(rr, dr=nr, rd=rn)
            print(f"Processing took {t2 - t1:.1f} seconds")            

        return nn


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


if __name__ == '__main__':
    PipelineStage.main()
