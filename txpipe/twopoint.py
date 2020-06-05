from .base_stage import PipelineStage
from .data_types import HDFFile, MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, PhotozPDFFile, PNGFile, TextFile
from .utils.metacal import apply_metacal_response
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


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('patch_centers', TextFile),
    ]
    outputs = [
        ('twopoint_data_real_raw', SACCFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':250.,
        'nbins':20,
        'bin_slop':0.1,
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
        'var_methods': 'jackknife',
        }

    def run(self):
        """
        Run the analysis for this stage.
        """
        import sacc
        import healpy
        import treecorr
        # Load the different pieces of data we need into
        # one large dictionary which we accumulate
        data = {}
        self.load_tomography(data)
        self.load_shear_catalog(data)
        self.load_random_catalog(data)
        # This one is optional - this class does nothing with it
        self.load_lens_catalog(data)
        # Binning information
        self.read_nbin(data)

        # Calculate metadata like the area and related
        # quantities
        meta = self.calculate_metadata(data)

        # Choose which pairs of bins to calculate
        calcs = self.select_calculations(data)

        sys.stdout.flush()

        # This splits the calculations among the parallel bins
        # It's not necessarily the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment,
        # but for this case I would expect it to be mostly fine
        results = []
        for i,j,k in self.split_tasks_by_rank(calcs):
            result = self.call_treecorr(data, i, j, k)
            results.append(result)

        # If we are running in parallel this collects the results together
        results = self.collect_results(results)

        # Save the results
        if self.rank==0:
            self.write_output(data, meta, results)


    def select_calculations(self, data):
        source_list = data['source_list']
        lens_list = data['lens_list']
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
            if not 'random_bin' in data:
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

    def read_nbin(self, data):
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

        data['source_list']  =  source_list
        data['lens_list']  =  lens_list


    # These two functions can be combined into a single one.
    def _read_nbin_from_tomography(self):
        tomo = self.open_input('shear_tomography_catalog')
        d = dict(tomo['tomography'].attrs)
        tomo.close()
        nbin_source = d['nbin_source']
        tomo = self.open_input('lens_tomography_catalog')
        d = dict(tomo['tomography'].attrs)
        tomo.close()
        nbin_lens = d['nbin_lens']
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



    def write_output(self, data, meta, results):
        import sacc
        import treecorr
        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        WTHETA = sacc.standard_types.galaxy_density_xi

        S = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data
        f = self.open_input('shear_photoz_stack')

        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        for i in data['source_list']:
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            S.add_tracer('NZ', f'source_{i}', z, Nz)

        f = self.open_input('lens_photoz_stack')
        # For both source and lens
        for i in data['lens_list']:
            z = f['n_of_z/lens/z'][:]
            Nz = f[f'n_of_z/lens/bin_{i}'][:]
            S.add_tracer('NZ', f'lens_{i}', z, Nz)
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
        cov = treecorr.estimate_multi_cov(comb, self.config['var_methods'])
        S.add_covariance(cov)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('twopoint_data_real_raw'), overwrite=True)

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


    def call_treecorr(self, data, i, j, k):
        """
        This is a wrapper for interaction with treecorr.
        """
        import sacc


        if k==SHEAR_SHEAR:
            xx = self.calculate_shear_shear(data, i, j)
            xtype = "combined"
        elif k==SHEAR_POS:
            xx = self.calculate_shear_pos(data, i, j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k==POS_POS:
            xx = self.calculate_pos_pos(data, i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        else:
            raise ValueError(f"Unknown correlation function {k}")

        result = Measurement(xtype, xx, i, j)

        sys.stdout.flush()
        return result


    def get_m(self, data, i):
        """
        Calculate the metacal correction factor for this tomographic bin.
        """

        mask = (data['source_bin'] == i)

        # We use S=0 here because we have already included it in R_total
        g1, g2 = apply_metacal_response(data['R_total'][i], 0.0, data['mcal_g1'][mask],data['mcal_g2'][mask])

        return g1, g2, mask


    def get_shear_catalog(self, data, i):
        import treecorr
        g1,g2,mask = self.get_m(data, i)

        if self.config['var_methods']=='jackknife':
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


    def get_lens_catalog(self, data, i):
        import treecorr


        mask = data['lens_bin'] == i

        if 'lens_ra' in data:
            ra = data['lens_ra'][mask]
            dec = data['lens_dec'][mask]
        else:
            ra = data['ra'][mask]
            dec = data['dec'][mask]

        if self.config['var_methods']=='jackknife':
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
            if self.config['var_methods']=='jackknife':
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


    def calculate_shear_shear(self, data, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(data, i)
        n_i = cat_i.nobj


        if i==j:
            cat_j = cat_i
            n_j = n_i
        else:
            cat_j = self.get_shear_catalog(data, j)
            n_j = cat_j.nobj


        print(f"Rank {self.rank} calculating shear-shear bin pair ({i},{j}): {n_i} x {n_j} objects")

        gg = treecorr.GGCorrelation(self.config)
        gg.process(cat_i, cat_j)

        return gg

    def calculate_shear_pos(self,data, i, j):
        import treecorr

        cat_i = self.get_shear_catalog(data, i)
        n_i = cat_i.nobj

        cat_j, rancat_j = self.get_lens_catalog(data, j)
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


    def calculate_pos_pos(self, data, i, j):
        import treecorr

        cat_i, rancat_i = self.get_lens_catalog(data, i)
        n_i = cat_i.nobj
        n_rand_i = rancat_i.nobj if rancat_i is not None else 0

        if i==j:
            cat_j = cat_i
            rancat_j = rancat_i
            n_j = n_i
            n_rand_j = n_rand_i
        else:
            cat_j, rancat_j = self.get_lens_catalog(data, j)
            n_j = cat_j.nobj
            n_rand_j = rancat_j.nobj if rancat_j is not None else 0

        print(f"Rank {self.rank} calculating position-position bin pair ({i},{j}): {n_i} x {n_j} objects, "
            f"{n_rand_i} x {n_rand_j} randoms")


        nn = treecorr.NNCorrelation(self.config)
        rn = treecorr.NNCorrelation(self.config)
        nr = treecorr.NNCorrelation(self.config)
        rr = treecorr.NNCorrelation(self.config)

        nn.process(cat_i,    cat_j)
        nr.process(cat_i,    rancat_j)
        rn.process(rancat_i, cat_j)
        rr.process(rancat_i, rancat_j)

        nn.calculateXi(rr, dr=nr, rd=rn)
        return nn

    def load_tomography(self, data):

        # Columns we need from the tomography catalog
        f = self.open_input('shear_tomography_catalog')
        source_bin = f['tomography/source_bin'][:]
        r_total = f['metacal_response/R_total'][:]
        f.close()

        f = self.open_input('lens_tomography_catalog')
        lens_bin = f['tomography/lens_bin'][:]
        f.close()

        data['source_bin']  =  source_bin
        data['lens_bin']  =  lens_bin
        data['R_total']  =  r_total

    def load_lens_catalog(self, data):
        # Subclasses can load an external lens catalog
        pass



    def load_shear_catalog(self, data):

        # Columns we need from the shear catalog
        cat_cols = ['ra', 'dec', 'mcal_g1', 'mcal_g2', 'mcal_flags']
        print(f"Loading shear catalog columns: {cat_cols}")

        f = self.open_input('shear_catalog')
        g = f['metacal']
        for col in cat_cols:
            print(f"Loading {col}")
            data[col] = g[col][:]

        if self.config['flip_g2']:
            data['mcal_g2'] *= -1


    def load_random_catalog(self, data):
        filename = self.get_input('random_cats')
        if filename is None:
            print("Not using randoms")
            return

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','e1','e2','ra','bin']
        print(f"Loading random catalog columns: {randoms_cols}")

        f = self.open_input('random_cats')
        group = f['randoms']

        cut = self.config['reduce_randoms_size']
        if 0.0<cut<1.0:
            N = group['dec'].size
            sel = np.random.uniform(size=N) < cut
        else:
            sel = slice(None)

        data['random_ra'] =  group['ra'][sel]
        data['random_dec'] = group['dec'][sel]
        data['random_e1'] =  group['e1'][sel]
        data['random_e2'] =  group['e2'][sel]
        data['random_bin'] = group['bin'][sel]

        f.close()

    def calculate_area(self, data):
        import healpy as hp
        pix=hp.ang2pix(4096, np.pi/2.-np.radians(data['dec']),np.radians(data['ra']), nest=True)
        area=hp.nside2pixarea(4096)*(180./np.pi)**2
        mask=np.bincount(pix)>0
        area=np.sum(mask)*area
        area=float(area) * 60. * 60.
        return area

    def calculate_metadata(self, data):
        tomo = self.open_input('shear_tomography_catalog')
        area = self.calculate_area(data)
        sigma_e = tomo['tomography/sigma_e'][:]
        N_eff = tomo['tomography/N_eff'][:]

        mean_g1_list = []
        mean_g2_list = []
        for i in data['source_list']:
            g1, g2, mask = self.get_m(data, i)
            mean_g1 = g1.mean()
            mean_g2 = g2.mean()
            mean_g1_list.append(mean_g1)
            mean_g2_list.append(mean_g2)

        meta = {}
        meta["neff"] =  N_eff
        meta["area"] =  area
        meta["sigma_e"] =  sigma_e
        meta["mean_e1"] =  mean_g1
        meta["mean_e2"] =  mean_g2

        return meta

class TXTwoPointLensCat(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint takes its
    lens sample from an external source instead of using
    the photometric sample.
    """
    name='TXTwoPointLensCat'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('lens_catalog', HDFFile),
        ('patch_centers', TextFile),
    ]
    def load_lens_catalog(self, data):
        filename = self.get_input('lens_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('lens_catalog')
        data['lens_ra']  = f['lens/ra'][:]
        data['lens_dec'] = f['lens/dec'][:]
        data['lens_bin'] = f['lens/bin'][:]


class TXTwoPointPlots(PipelineStage):
    """
    Make n(z) plots

    """
    name='TXTwoPointPlots'
    inputs = [
        ('twopoint_data_real', SACCFile),
    ]
    outputs = [
        ('shear_xi', PNGFile),
        ('shear_xi_err', PNGFile),
        ('shearDensity_xi', PNGFile),
        ('shearDensity_xi_err', PNGFile),
        ('density_xi', PNGFile),
        ('density_xi_err', PNGFile),
    ]

    config_options = {
        'wspace': 0.1,
        'hspace': 0.1,
    }


    def run(self):
        import sacc
        import matplotlib
        matplotlib.use('agg')
        matplotlib.rcParams["xtick.direction"]='in'
        matplotlib.rcParams["ytick.direction"]='in'

        gammat = sacc.standard_types.galaxy_shearDensity_xi_t
        wtheta = sacc.standard_types.galaxy_density_xi

        filename = self.get_input('twopoint_data_real')
        s = sacc.Sacc.load_fits(filename)

        sources, lenses = self.read_nbin(s)
        print(f"Plotting xi for {len(sources)} sources and {len(lenses)} lenses")

        self.colors = ['steelblue', 'orange']
        self.plot_shear_shear(s, sources)
        self.plot_shear_density(s, sources, lenses)
        self.plot_density_density(s, lenses)

        
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



class TXGammaTFieldCenters(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of exposure fields as "lenses", as a systematics test.
    """
    name = "TXGammaTFieldCenters"
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('exposures', HDFFile),
        ('patch_centers', TextFile),
    ]
    outputs = [
        ('gammat_field_center', SACCFile),
        ('gammat_field_center_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':250,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        'var_methods': 'jackknife',
        'npatch': 5
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        filename = self.get_input('exposures')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('exposures')
        data['lens_ra']  = f['exposures/ratel'][:]
        data['lens_dec'] = f['exposures/dectel'][:]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output('gammat_field_center_plot', wrapper=True)

        plt.errorbar(dtheta,  dtheta*dvalue, derror, fmt='ro', capsize=3)
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Field Center Tangential Shear")

        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyFieldCenter_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('shear_photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'fieldcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'fieldcenter'), dvalue[i],
                theta=dtheta[i], error=derror[i], npair=dnpair[i], weight=dweight[i])

        #self.write_metadata(S, meta)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_field_center'), overwrite=True)

        # Also make a plot of the data

class TXGammaTBrightStars(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """
    name = "TXGammaTBrightStars"
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('star_catalog', HDFFile),
        ('patch_centers', TextFile),
    ]
    outputs = [
        ('gammat_bright_stars', SACCFile),
        ('gammat_bright_stars_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':100,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        'var_methods': 'shot',
        'npatch': 5
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        # TODO break up bright and dim stars
        #14<mi <18.3forthebrightsampleand18.3<mi <22 in DES
        filename = self.get_input('star_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('star_catalog')

        mags = f['stars/r_mag'][:]
        bright_cut = mags>14
        bright_cut &= mags<18.3

        data['lens_ra']  = f['stars/ra'][:][bright_cut]
        data['lens_dec'] = f['stars/dec'][:][bright_cut]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output('gammat_bright_stars_plot', wrapper=True)

        # compute the mean and the chi^2/dof
        z = (dvalue) / derror
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(dtheta) - 1)
        print('error,',derror)

        plt.errorbar(dtheta,  dtheta*dvalue, dtheta*derror, fmt='ro', capsize=3,label='$\chi^2/dof = $'+str(chi2dof))
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Bright Star Centers Tangential Shear")

        print('type',type(fig))
        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyStarCenters_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('shear_photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'starcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight

        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'starcenter'), dvalue[i],
                theta=dtheta[i], error=derror[i], npair=dnpair[i], weight=dweight[i])

        self.write_metadata(S, meta)

        print(S)
        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_bright_stars'), overwrite=True)

        # Also make a plot of the data

class TXGammaTDimStars(TXTwoPoint):
    """
    This subclass of the standard TXTwoPoint uses the centers
    of stars as "lenses", as a systematics test.
    """
    name = "TXGammaTDimStars"
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog),
        ('star_catalog', HDFFile),
        ('patch_centers', TextFile),
    ]
    outputs = [
        ('gammat_dim_stars', SACCFile),
        ('gammat_dim_stars_plot', PNGFile),
    ]
    # Add values to the config file that are not previously defined
    config_options = {
        'calcs':[0,1,2],
        'min_sep':2.5,
        'max_sep':100,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'reduce_randoms_size':1.0,
        'var_methods': 'jackknife',
        'npatch': 5
        }

    def run(self):
        # Before running the parent class we add source_bins and lens_bins
        # options that it is expecting, both set to -1 to indicate that we
        # will choose them automatically (below).
        import matplotlib
        matplotlib.use('agg')
        self.config['source_bins'] = [-1]
        self.config['lens_bins'] = [-1]
        super().run()

    def read_nbin(self, data):
        # We use only a single source and lens bin in this case -
        # the source is the complete 2D field and the lens is the
        # field centers
        data['source_list'] = [0]
        data['lens_list'] = [0]

    def load_lens_catalog(self, data):
        # We load our lenses from the exposures input.
        # TODO break up bright and dim stars
        #14<mi <18.3forthebrightsampleand18.3<mi <22 in DES
        filename = self.get_input('star_catalog')
        print(f"Loading lens sample from {filename}")

        f = self.open_input('star_catalog')
        mags = f['stars/r_mag'][:]
        dim_cut = mags>18.2
        dim_cut &= mags<22

        data['lens_ra']  = f['stars/ra'][:][dim_cut]
        data['lens_dec'] = f['stars/dec'][:][dim_cut]
        f.close()

        npoint = data['lens_ra'].size
        data['lens_bin'] = np.zeros(npoint)

    def load_tomography(self, data):
        # We run the parent class tomography selection but then
        # overrided it to squash all of the bins  0 .. nbin -1
        # down to the zero bin.  This means that any selected
        # objects (from any tomographic bin) are now in the same
        # bin, and unselected objects still have bin -1
        super().load_tomography(data)
        data['source_bin'][:] = data['source_bin'].clip(-1,0)

    def select_calculations(self, data):
        # We only want a single calculation, the gamma_T around
        # the field centers
        return [(0,0,SHEAR_POS)]

    def write_output(self, data, meta, results):
        # we write output both to file for later and to
        # a plot
        self.write_output_sacc(data, meta, results)
        self.write_output_plot(results)

    def write_output_plot(self, results):
        import matplotlib.pyplot as plt
        d = results[0]
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)

        fig = self.open_output('gammat_dim_stars_plot', wrapper=True)

        # compute the mean and the chi^2/dof
        flat1 = 0
        z = (dvalue - flat1) / derror
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (len(dtheta) - 1)
        print('error,',derror)

        plt.errorbar(dtheta,  dtheta*dvalue, dtheta*derror, fmt='ro', capsize=3,label='$\chi^2/dof = $'+str(chi2dof))
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r"$\theta$ / arcmin")
        plt.ylabel(r"$\theta \cdot \gamma_t(\theta)$")
        plt.title("Dim Star Centers Tangential Shear")

        print('type',type(fig))
        fig.close()

    def write_output_sacc(self, data, meta, results):
        # We write out the results slightly differently here
        # beause they go to a different file and have different
        # tracers and tags.
        import sacc
        dt = "galaxyStarCenters_shearDensity_xi_t"

        S = sacc.Sacc()

        f = self.open_input('shear_photoz_stack')
        z = f['n_of_z/source2d/z'][:]
        Nz = f[f'n_of_z/source2d/bin_0'][:]
        f.close()

        # Add the data points that we have one by one, recording which
        # tracer they each require
        S.add_tracer('misc', 'starcenter')
        S.add_tracer('NZ', 'source2d', z, Nz)

        d = results[0]
        assert len(results)==1
        dvalue = d.object.xi
        derror = np.sqrt(d.object.varxi)
        dtheta = np.exp(d.object.meanlogr)
        dnpair = d.object.npairs
        dweight = d.object.weight


        # Each of our Measurement objects contains various theta values,
        # and we loop through and add them all
        n = len(dvalue)
        for i in range(n):
            S.add_data_point(dt, ('source2d', 'starcenter'), dvalue[i],
                theta=dtheta[i], error=derror[i], npair=dnpair[i], weight=dweight[i])

        self.write_metadata(S, meta)

        print(S)
        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('gammat_dim_stars'), overwrite=True)

        # Also make a plot of the data

class TXJackknifeCenters(PipelineStage):
    """
    This is the pipeline stage that is run to generate the patch centers for
    the Jackknife method.
    """
    name = 'TXJackknifeCenters'

    inputs = [
        ('random_cats', RandomsCatalog),
    ]
    outputs = [
        ('patch_centers', TextFile),
        ('jk', PNGFile),
    ]
    config_options = {
        'npatch' : 10,
    }

    def plot(self, ra, dec, patch):
        """
        Plot the jackknife regions.
        """
        import matplotlib
        matplotlib.use('agg')
        matplotlib.rcParams["xtick.direction"]='in'
        matplotlib.rcParams["ytick.direction"]='in'
        import matplotlib.pyplot as plt

        print(ra, dec, patch)

        jk_plot = self.open_output('jk', wrapper=True, figsize=(6.,4.5))
        # Choose colormap
        cm = plt.cm.get_cmap('magma')
        sc = plt.scatter(ra, dec, c = patch, cmap = cm,  s=20, vmin = 0)
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.tight_layout()
        jk_plot.close()


    def run(self):
        import treecorr

        filename = self.get_input('random_cats')
        if filename is None:
            print("Not using randoms, we need to use randoms for now")
            return

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','ra']
        print(f"Loading random catalog columns: {randoms_cols}")

        f = self.open_input('random_cats')
        group = f['randoms']
        npatch=self.config['npatch']
        print(f"generating {npatch} centers")
        ra = group['ra'][:]
        dec = group['dec'][:]
        cat = treecorr.Catalog(ra = ra,
                                dec = dec,
                                ra_units='degree', dec_units = 'degree',
                                npatch=self.config['npatch'])
        cat.write_patch_centers(self.get_output('patch_centers'))

        self.plot(cat.ra, cat.dec, cat.patch)


class TXSelfCalibrationIA(TXTwoPoint):
    """
    This is the subclass of the Twopoint class that will handle calculating the
    correlations needed for doing the Self-calibration Intrinsic alignment
    estimation. 
    """
    name = 'TXSCIA'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('shear_photoz_stack', HDFFile),
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_photoz_stack', HDFFile),
        ('random_cats_source', RandomsCatalog),
        ('patch_centers', TextFile),
        ('photoz_pdfs', PhotozPDFFile),
        ('fiducial_cosmology', YamlFile),
    ]
    outputs = [
        ('twopoint_data', SACCFile),
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
        'do_pos_pos': True,
        'var_methods': 'jackknife',
        '3Dcoords': True,
        'metric': 'Rperp'
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
        
        if self.rank==0:
            print(f"Running these calculations: {calcs}")
        
        return calcs

    def load_shear_catalog(self, data):

        # Columns we need from the shear catalog
        cat_cols = ['ra', 'dec', 'mcal_g1', 'mcal_g2', 'mcal_flags']
        print(f"Loading shear catalog columns: {cat_cols}")

        f = self.open_input('shear_catalog')
        g = f['metacal']
        for col in cat_cols:
            print(f"Loading {col}")
            data[col] = g[col][:]

        if self.config['flip_g2']:
            data['mcal_g2'] *= -1   

        if self.config['3Dcoords']:
            h = self.open_input('photoz_pdfs')
            g = h['pdf']
            data['mu'] = g['mu'][:]

    def load_random_catalog(self, data):
        filename = self.get_input('random_cats_source')
        if filename is None:
            print("Not using randoms")
            return

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','e1','e2','ra','bin','z']
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
        data['random_e1'] =  group['e1'][sel]
        data['random_e2'] =  group['e2'][sel]
        data['random_bin'] = group['bin'][sel]
        data['random_z'] = group['z'][sel]

        f.close()

    def get_lens_catalog(self, data, i):
        import treecorr
        import pyccl as ccl

        #Note that we are here actually loading the source bin as the lens bin!
        #mask = data['source_bin'] == i
        g1,g2,mask = self.get_m(data, i)

        
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


            if self.config['var_methods']=='jackknife':
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
                if self.config['var_methods']=='jackknife':
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
            if self.config['var_methods']=='jackknife':
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
                if self.config['var_methods']=='jackknife':
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

    def get_shear_catalog(self, data, i):
        import treecorr
        import pyccl as ccl
        g1,g2,mask = self.get_m(data, i)
        
        if self.config['3Dcoords']:
            mu = data['mu'][mask]
            cosmo = ccl.Cosmology.read_yaml(self.get_input('fiducial_cosmology'))
            r = ccl.background.comoving_radial_distance(cosmo, 1/(1+mu))

            if self.config['var_methods']=='jackknife':
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
            if self.config['var_methods']=='jackknife':
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

    def calculate_shear_pos_select(self,data, i, j):
        import treecorr 

        cat_i = self.get_shear_catalog(data, i)
        n_i = cat_i.nobj

        cat_j, rancat_j = self.get_lens_catalog(data, j)
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

    def call_treecorr(self, data, i, j, k):
        import sacc 

        if k==SHEAR_SHEAR:
            xx = self.calculate_shear_shear(data, i, j)
            xtype = "combined"
        elif k==SHEAR_POS:
            xx = self.calculate_shear_pos(data, i, j)
            xtype = sacc.standard_types.galaxy_shearDensity_xi_t
        elif k==SHEAR_POS_SELECT:
            xx = self.calculate_shear_pos_select(data, i, j)
            xtype = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='ts')
        elif k==POS_POS:
            xx = self.calculate_pos_pos(data, i, j)
            xtype = sacc.standard_types.galaxy_density_xi
        else:
            raise ValueError(f"Unknown correlation function {k}")

    def write_output(self, data, meta, results):
        import sacc
        import treecorr
        XI = "combined"
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.galaxy_shearDensity_xi_t
        GAMMATS = sacc.build_data_type_name('galaxy',['shear','Density'],'xi',subtype ='ts')
        WTHETA = sacc.standard_types.galaxy_density_xi

        S = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data
        f = self.open_input('shear_photoz_stack')

        # Load the tracer data N(z) from an input file and
        # copy it to the output, for convenience
        for i in data['source_list']:
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            S.add_tracer('NZ', f'source_{i}', z, Nz)

        f = self.open_input('lens_photoz_stack')
        # For both source and lens
        for i in data['lens_list']:
            z = f['n_of_z/lens/z'][:]
            Nz = f[f'n_of_z/lens/bin_{i}'][:]
            S.add_tracer('NZ', f'lens_{i}', z, Nz)
        # Closing n(z) file
        f.close()

        # Now build up the collection of data points, adding them all to
        # the sacc data one by one.
        comb = []
        for d in results:
            # First the tracers and generic tags
            tracer1 = f'source_{d.i}' if d.corr_type in [XI, GAMMAT,GAMMATS] else f'lens_{d.i}'
            tracer2 = f'source_{d.j}' if d.corr_type in [XI, GAMMAT, GAMMATS] else f'lens_{d.j}'

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
        cov = treecorr.estimate_multi_cov(comb, self.config['var_methods'])
        S.add_covariance(cov)

        # Our data points may currently be in any order depending on which processes
        # ran which calculations.  Re-order them.
        S.to_canonical_order()

        # Finally, save the output to Sacc file
        S.save_fits(self.get_output('twopoint_data_real_raw'), overwrite=True)


if __name__ == '__main__':
    PipelineStage.main()
