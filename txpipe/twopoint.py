from ceci import PipelineStage
from .data_types import HDFFile, MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, PhotozPDFFile
import numpy as np
import random
import collections
import sys

# This creates a little mini-type, like a struct,
# for holding individual measurements
Measurement = collections.namedtuple(
    'Measurement',
    ['corr_type', 'theta', 'value', 'error', 'npair', 'weight', 'i', 'j'])

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2



class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('photoz_stack', HDFFile),
        ('random_cats', RandomsCatalog)
    ]
    outputs = [
        ('twopoint_data', SACCFile),
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
        'source_bins':[-1],
        'lens_bins':[-1],
        'reduce_randoms_size':1.0,
        }

    def run(self):
        """
        Run the analysis for this stage.

         - reads the config file
         - prepares the output HDF5 file
         - loads in the data
         - computes 3x2 pt. correlation functions
         - writes the to output
         - closes the output file

        """

        # Load the different pieces of data we need into
        # one large dictionary which we accumulate
        data = {}
        self.load_tomography(data)
        self.load_shear_catalog(data)
        self.load_random_catalog(data)

        # Calculate metadata like the area and related
        # quantities
        meta = self.calc_metadata(data)

        # Choose which pairs of bins to calculate
        calcs = self.select_calculations(data)

        # This splits the calculations among the parallel bins
        # It's not necessarily the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment,
        # but for this case I would expect it to be mostly fine
        results = []
        for i,j,k in self.split_tasks_by_rank(calcs):
            results += self.call_treecorr(data, i, j, k)

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
        k = SHEAR_SHEAR
        for i in source_list:
            for j in range(i+1):
                if j in source_list:
                    calcs.append((i,j,k))

        # For shear-position we use all pairs
        k = SHEAR_POS
        for i in source_list:
            for j in lens_list:
                calcs.append((i,j,k))

        # For position-position we omit pairs with j>i
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

    def read_nbin(self, data):
        """
        Determine the bins that
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



    def _read_nbin_from_tomography(self):

        tomo = self.open_input('tomography_catalog')
        d = dict(tomo['tomography'].attrs)
        tomo.close()
        nbin_source = d['nbin_source']
        nbin_lens = d['nbin_lens']
        source_list = range(nbin_source)
        lens_list = range(nbin_lens)
        return source_list, lens_list

    def _read_nbin_from_config(self):
        # TODO handle the case where the user only specefies 
        # bins for only sources or only lenses
        source_list = self.config['source_bins']
        lens_list = self.config['lens_bins']
        nbin_source = len(source_list)
        nbin_lens = len(lens_list)

        # catch bad input
        tomo_source_list, tomo_lens_list = self._read_nbin_from_tomography()
        tomo_nbin_source = len(tomo_source_list)
        tomo_nbin_lens = len(tomo_lens_list)
        # if more bins are input than exist, assertion error
        assert (nbin_source <= tomo_nbin_source), 'too many source bins, entered {} max is {}'.format(nbin_source, tom_nbin_source)
        assert (nbin_lens <= tomo_nbin_lens), 'too many lens bins, entered {} max is {}'.format(nbin_lens, tom_nbin_lens)
        # make sure the bin numbers actually exist
        for i in source_list:
            assert i in range(tomo_nbin_source), 'source bin {i} is out of bounds'
        for j in lens_list:
            assert j in range(tomo_nbin_lens), 'lens bin {j} is out of bounds'
            
        return source_list, lens_list 



    def write_output(self, data, meta, results):
        import sacc
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.ggl_gamma_t

        S = sacc.Sacc()

        # We include the n(z) data in the output.
        # So here we load it in and add it to the data
        f = self.open_input('photoz_stack')

        # F
        for i in data['source_list']:
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            S.add_tracer('NZ', f'source_{i}', z, Nz)

        for i in data['lens_list']:
            z = f['n_of_z/lens/z'][:]
            Nz = f[f'n_of_z/lens/bin_{i}'][:]
            S.add_tracer('NZ', f'lens_{i}', z, Nz)

        f.close()

        for d in results:
            tracer1 = f'source_{d.i}' if d.corr_type in [XIP, XIM, GAMMAT] else f'lens_{d.i}'
            tracer2 = f'source_{d.j}' if d.corr_type in [XIP, XIM] else f'lens_{d.j}'
            n = len(d.value)
            for i in range(n):
                S.add_data_point(d.corr_type, (tracer1,tracer2), d.value[i], 
                    theta=d.theta[i], error=d.error[i], npair=d.npair[i], weight=d.weight[i])

        for k,v in meta.items():
            if np.isscalar(v):
                S.metadata[k] = v
            else:
                for i, vi in enumerate(v):
                    S.metadata[f'{k}_{i}'] = vi
        S.to_canonical_order()
        S.save_fits(self.get_output('twopoint_data'), overwrite=True)



    def call_treecorr(self, data, i, j, k):
        """
        This is a wrapper for interaction with treecorr.
        """
        import sacc
        XIP = sacc.standard_types.galaxy_shear_xi_plus
        XIM = sacc.standard_types.galaxy_shear_xi_minus
        GAMMAT = sacc.standard_types.ggl_gamma_t
        WTHETA = sacc.standard_types.galaxy_density_w

        results = []

        if k==SHEAR_SHEAR:
            theta, xip, xim, xiperr, ximerr, npairs, weight = self.calc_shear_shear(data, i, j)
            if i==j:
                npairs/=2
                weight/=2

            results.append(Measurement(XIP, theta, xip, xiperr, npairs, weight, i, j))
            results.append(Measurement(XIM, theta, xim, ximerr, npairs, weight, i, j))

        elif k==SHEAR_POS:
            theta, val, err, npairs, weight = self.calc_shear_pos(data, i, j)
            if i==j:
                npairs/=2
                weight/=2

            results.append(Measurement(GAMMAT, theta, val, err, npairs, weight, i, j))

        elif k==POS_POS:
            theta, val, err, npairs, weight = self.calc_pos_pos(data, i, j)
            if i==j:
                npairs/=2
                weight/=2

            results.append(Measurement(WTHETA, theta, val, err, npairs, weight, i, j))

        return results


    def get_m(self, data, i):

        mask = (data['source_bin'] == i)

        m1 = np.mean(data['r_gamma'][mask][:,0,0]) # R11, taking the mean for the bin, TODO check if that's what we want to do
        m2 = np.mean(data['r_gamma'][mask][:,1,1]) #R22

        return m1, m2, mask

    def calc_shear_shear(self, data, i, j):
        import treecorr
        m1,m2,mask = self.get_m(data, i)

        g1 = data['mcal_g1'][mask]
        g2 = data['mcal_g2'][mask]
        n_i = len(g1)

        cat_i = treecorr.Catalog(
            g1 = (g1 - g1.mean()) / m1,
            g2 = (g2 - g2.mean()) / m2,
            ra=data['ra'][mask], dec=data['dec'][mask],
            ra_units='degree', dec_units='degree')

        del g1, g2

        if i==j:
            cat_j = cat_i
            n_j = n_i
        else:
            m1,m2,mask = self.get_m(data, j)
            g1 = data['mcal_g1'][mask]
            g2 = data['mcal_g2'][mask]
            n_j = len(g1)

            cat_j = treecorr.Catalog(
                g1 = (g1 - g1.mean()) / m1,
                g2 = (g2 - g2.mean()) / m2,
                ra=data['ra'][mask], dec=data['dec'][mask],
                ra_units='degree', dec_units='degree')
            del g1, g2

        print(f"Rank {self.rank} calculating shear-shear bin pair ({i},{j}): {n_i} x {n_j} objects")

        gg = treecorr.GGCorrelation(self.config)
        gg.process(cat_i, cat_j)

        theta=np.exp(gg.meanlogr)
        xip = gg.xip
        xim = gg.xim
        xiperr = np.sqrt(gg.varxip)
        ximerr = np.sqrt(gg.varxim)

        return theta, xip, xim, xiperr, ximerr, gg.npairs, gg.weight

    def calc_shear_pos(self,data, i, j):
        import treecorr

        m1,m2,mask = self.get_m(data, i)

        g1 = data['mcal_g1'][mask]
        g2 = data['mcal_g2'][mask]
        n_i = len(g1)
        cat_i = treecorr.Catalog(
            g1 = (g1 - g1.mean()) / m1,
            g2 = (g2 - g2.mean()) / m2,
            ra = data['ra'][mask],
            dec = data['dec'][mask],
            ra_units='degree', dec_units='degree')
        del g1, g2

        mask = data['lens_bin'] == j
        random_mask = data['random_bin'] == j
        n_j = mask.sum()
        n_rand = random_mask.sum()

        cat_j = treecorr.Catalog(
            ra=data['ra'][mask],
            dec=data['dec'][mask],
            ra_units='degree',
            dec_units='degree')

        rancat_j  = treecorr.Catalog(
            ra=data['random_ra'][random_mask],
            dec=data['random_dec'][random_mask],
            ra_units='degree',
            dec_units='degree')

        print(f"Rank {self.rank} calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand} randoms")

        ng = treecorr.NGCorrelation(self.config)
        rg = treecorr.NGCorrelation(self.config)

        ng.process(cat_j, cat_i)
        rg.process(rancat_j, cat_i)

        gammat, gammat_im, gammaterr = ng.calculateXi(rg=rg)

        theta = np.exp(ng.meanlogr)
        gammaterr = np.sqrt(gammaterr)

        return theta, gammat, gammaterr, ng.npairs, ng.weight

    def calc_pos_pos(self, data, i, j):
        import treecorr

        mask = data['lens_bin'] == i
        random_mask = data['random_bin']==i
        n_i = mask.sum()
        n_rand_i = random_mask.sum()

        cat_i = treecorr.Catalog(
            ra=data['ra'][mask], dec=data['dec'][mask],
            ra_units='degree', dec_units='degree')

        rancat_i  = treecorr.Catalog(
            ra=data['random_ra'][random_mask], dec=data['random_dec'][random_mask],
            ra_units='degree', dec_units='degree')

        if i==j:
            cat_j = cat_i
            rancat_j = rancat_i
            n_j = n_i
            n_rand_j = n_rand_i
        else:
            mask = data['lens_bin'] == j
            random_mask = data['random_bin'] == j
            n_j = mask.sum()
            n_rand_j = random_mask.sum()

            cat_j = treecorr.Catalog(
                ra=data['ra'][mask], dec=data['dec'][mask],
                ra_units='degree', dec_units='degree')

            rancat_j  = treecorr.Catalog(
                ra=data['random_ra'][random_mask], dec=data['random_dec'][random_mask],
                ra_units='degree', dec_units='degree')

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

        theta=np.exp(nn.meanlogr)
        wtheta,wthetaerr=nn.calculateXi(rr, dr=nr, rd=rn)
        wthetaerr=np.sqrt(wthetaerr)

        return theta, wtheta, wthetaerr, nn.npairs, nn.weight

    def load_tomography(self, data):

        # Columns we need from the tomography catalog
        f = self.open_input('tomography_catalog')
        source_bin = f['tomography/source_bin'][:]
        lens_bin = f['tomography/lens_bin'][:]
        f.close()

        f = self.open_input('tomography_catalog')
        r_gamma = f['multiplicative_bias/R_gamma'][:]
        f.close()

        self.read_nbin(data)

        data['source_bin']  =  source_bin
        data['lens_bin']  =  lens_bin
        data['r_gamma']  =  r_gamma
        

    def load_shear_catalog(self, data):

        # Columns we need from the shear catalog
        cat_cols = ['ra', 'dec', 'mcal_g1', 'mcal_g2', 'mcal_flags',]
        # JAZ I couldn't see a use for these at the moment - will probably need them later,
        # though may be able to do those algorithms on-line
        # cat_cols += ['mcal_mag','mcal_s2n_r', 'mcal_T']

        print(f"Loading shear catalog columns: {cat_cols}")

        f = self.open_input('shear_catalog')
        g = f['metacal']
        for col in cat_cols:
            print(f"Loading {col}")
            data[col] = g[col][:]

        if self.config['flip_g2']:
            data['mcal_g2'] *= -1


    def load_random_catalog(self, data):

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


    def calc_sigma_e(self, data):
        """
        Calculate sigma_e for shape catalog.
        """
        sigma_e_list = []
        mean_g1_list = []
        mean_g2_list = []
        for i in data['source_list']:
            m1, m2, mask = self.get_m(data, i)
            s = (m1+m2)/2
            g1 = data['mcal_g1'][mask]
            g2 = data['mcal_g2'][mask]
            mean_g1 = g1.mean()
            mean_g2 = g2.mean()
            # TODO Placeholder for actual weights we want to use
            w = np.ones_like(g1)
            a1 = np.sum(w**2 * (g1-mean_g1)**2)
            a2 = np.sum(w**2 * (g2-mean_g2)**2)
            b  = np.sum(w**2)
            c  = np.sum(w*s)
            d  = np.sum(w)

            sigma_e = np.sqrt( (a1/c**2 + a2/c**2) * (d**2/b) / 2. )

            sigma_e_list.append(sigma_e)
            mean_g1_list.append(mean_g1)
            mean_g2_list.append(mean_g2)

        return sigma_e_list, mean_g1_list, mean_g2_list

    def calc_neff(self, area, data):
        neff = []
        for i in data['source_list']:
            m1, m2, mask = self.get_m(data, i)
            w    = np.ones(len(data['ra'][mask]))
            a    = np.sum(w)**2
            b    = np.sum(w**2)
            c    = area
            neff.append(a/b/c)
        return neff

    def calc_area(self, data):
        import healpy as hp
        pix=hp.ang2pix(4096, np.pi/2.-np.radians(data['dec']),np.radians(data['ra']), nest=True)
        area=hp.nside2pixarea(4096)*(180./np.pi)**2
        mask=np.bincount(pix)>0
        area=np.sum(mask)*area
        area=float(area) * 60. * 60.
        return area

    def calc_metadata(self, data):
        #TODO put the metadata in the output SACC file
        area = self.calc_area(data)
        neff = self.calc_neff(area, data)
        sigma_e, mean_e1, mean_e2 = self.calc_sigma_e(data)

        meta = {}
        meta["neff"] =  neff
        meta["area"] =  area
        meta["sigma_e"] =  sigma_e
        meta["mean_e1"] =  mean_e1
        meta["mean_e2"] =  mean_e2

        return meta

if __name__ == '__main__':
    PipelineStage.main()
