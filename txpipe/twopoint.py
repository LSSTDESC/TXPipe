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

        if self.comm:
            self.comm.Barrier()

        # Get the number of bins from the
        # tomography input file
        nbins_source, nbins_lens  = self.read_nbins()

        print(f'Running with {nbins_source} source bins and {nbins_lens} lens bins')

        # Various setup and input functions.
        self.load_tomography()
        self.load_shear_catalog()
        self.load_random_catalog()
        self.setup_results()
        meta = self.calc_metadata(nbins_source)

        calcs = self.select_calculations(nbins_source, nbins_lens)
        if self.rank==0:
            print(f"Running these calculations: {calcs}")
        # This splits the calculations among the parallel bins
        # It's not necessarily the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment,
        # but for this case I would expect it to be mostly finer
        for i,j,k in self.split_tasks_by_rank(calcs):
            self.call_treecorr(i, j, k)

        self.collect_results()

        # Prepare the HDF5 output.
        # When we do this in parallel we can probably
        # just copy all the results to the root process
        # to output
        if self.rank==0:
            self.write_output(nbins_source, nbins_lens, meta)


    def select_calculations(self, nbins_source, nbins_lens):
        calcs = []

        # For shear-shear we omit pairs with j<i
        k = SHEAR_SHEAR
        for i in range(nbins_source):
            for j in range(i+1):
                calcs.append((i,j,k))

        # For shear-position we use all pairs
        k = SHEAR_POS
        for i in range(nbins_source):
            for j in range(nbins_lens):
                calcs.append((i,j,k))

        # For position-position we omit pairs with j<i
        k = POS_POS
        for i in range(nbins_lens):
            for j in range(i+1):
                calcs.append((i,j,k))

        return calcs

    def collect_results(self):
        if self.comm is None:
            return

        self.results = self.comm.gather(self.results, root=0)

        if self.rank==0:
            # Concatenate results
            self.results = sum(self.results, [])

            # Order by type, then bin i, then bin j
            order = [b'xip', b'xim', b'gammat', b'wtheta']
            key = lambda r: (order.index(r.corr_type), r.i, r.j)
            self.results = sorted(self.results, key=key)

    def setup_results(self):
        self.results = []


    def read_nbins(self):
        tomo = self.open_input('tomography_catalog')
        d = dict(tomo['tomography'].attrs)
        tomo.close()
        nbin_source = d['nbin_source']
        nbin_lens = d['nbin_lens']
        return nbin_source, nbin_lens

    def write_output(self, nbins_source, nbins_lens, meta):
        import sacc
        # TODO fix this to account for the case where we only do a certain number of calcs
        f = self.open_input('photoz_stack')

        tracers = []

        for i in range(nbins_source):
            z = f['n_of_z/source/z'].value
            Nz = f[f'n_of_z/source/bin_{i}'].value
            T=sacc.Tracer(f"LSST source_{i}","spin0", z, Nz, exp_sample="LSST-source")
            tracers.append(T)

        for i in range(nbins_lens):
            z = f['n_of_z/lens/z'].value
            Nz = f[f'n_of_z/lens/bin_{i}'].value
            T=sacc.Tracer(f"LSST lens_{i}","spin0", z, Nz, exp_sample="LSST-lens")
            tracers.append(T)


        f.close()

        fields = ['corr_type', 'theta', 'value', 'error', 'npair', 'weight', 'i', 'j']
        output = {f:list() for f in fields}

        q1 = []
        q2 = []
        type = []
        for corr_type in [b'xip', b'xim', b'gammat', b'wtheta']:
            data = [r for r in self.results if r.corr_type==corr_type]
            for bin_pair_data in data:
                n = len(bin_pair_data.theta)
                type+=['Corr' for i in range(n)]

                if corr_type==b'gammat':
                    q1 += ['P' for i in range(n)]
                    q2 += ['S' for i in range(n)]
                elif corr_type==b'wtheta':
                    q1 += ['P' for i in range(n)]
                    q2 += ['P' for i in range(n)]
                elif corr_type==b'xip':
                    q1 += ['+' for i in range(n)]
                    q2 += ['+' for i in range(n)]
                elif corr_type==b'xim':
                    q1 += ['-' for i in range(n)]
                    q2 += ['-' for i in range(n)]
                for f in fields:
                    v = getattr(bin_pair_data, f)
                    if np.isscalar(v):
                        v = [v for i in range(n)]
                    else:
                        v = v.tolist()
                    output[f] += v

        binning=sacc.Binning(type,output['theta'],output['i'],q1,output['j'],q2)
        mean=sacc.MeanVec(output['value'])
        s=sacc.SACC(tracers,binning,mean,meta=meta)
        s.printInfo()
        output_filename = self.get_output("twopoint_data")
        s.saveToHDF(output_filename)

    def call_treecorr(self,i,j,k):
        """
        This is a wrapper for interaction with treecorr.
        """
        # k==0: xi+-
        # k==1: gammat
        # k==2: wtheta

        # Cori value


        if k==SHEAR_SHEAR: # xi+-
            theta, xip, xim, xiperr, ximerr, npairs, weight = self.calc_shear_shear(i,j)
            if i==j:
                npairs/=2
                weight/=2

            self.results.append(Measurement(b"xip", theta, xip, xiperr, npairs, weight, i, j))
            self.results.append(Measurement(b"xim", theta, xim, ximerr, npairs, weight, i, j))

        elif k==SHEAR_POS: # gammat
            theta, val, err, npairs, weight = self.calc_shear_pos(i,j)
            if i==j:
                npairs/=2
                weight/=2

            self.results.append(Measurement(b"gammat", theta, val, err, npairs, weight, i, j))

        elif k==POS_POS: # wtheta
            theta, val, err, npairs, weight = self.calc_pos_pos(i,j)
            if i==j:
                npairs/=2
                weight/=2

            self.results.append(Measurement(b"wtheta", theta, val, err, npairs, weight, i, j))



    def get_m(self,i):

        mask = (self.binning == i)

        m1 = np.mean(self.r_gamma[mask][:,0,0]) # R11, taking the mean for the bin, TODO check if that's what we want to do
        m2 = np.mean(self.r_gamma[mask][:,1,1]) #R22

        return m1, m2, mask

    def calc_shear_shear(self,i,j):
        import treecorr
        m1,m2,mask = self.get_m(i)
        g1 = self.mcal_g1[mask]
        g2 = self.mcal_g2[mask]
        n_i = len(g1)

        cat_i = treecorr.Catalog(
            g1 = (g1 - g1.mean()) / m1,
            g2 = (g2 - g2.mean()) / m2,
            ra=self.ra[mask], dec=self.dec[mask],
            ra_units='degree', dec_units='degree')
        del g1, g2

        if i==j:
            cat_j = cat_i
            n_j = n_i
        else:

            m1,m2,mask = self.get_m(j)
            g1 = self.mcal_g1[mask]
            g2 = self.mcal_g2[mask]
            n_j = len(g1)

            cat_j = treecorr.Catalog(
                g1 = (g1 - g1.mean()) / m1,
                g2 = (g2 - g2.mean()) / m2,
                ra=self.ra[mask], dec=self.dec[mask],
                ra_units='degree', dec_units='degree')
            del g1, g2

        print(f"Rank {self.rank} calculating shear-shear bin pair ({i},{j}): {n_i} x {n_j} objects")

        gg = treecorr.GGCorrelation(self.config)
        gg.process(cat_i, cat_j)

        theta=np.exp(gg.meanlogr)
        xip = gg.xip
        xim = gg.xim
        xiperr = ximerr = np.sqrt(gg.varxi)

        return theta, xip, xim, xiperr, ximerr, gg.npairs, gg.weight

    def calc_shear_pos(self,i,j):
        import treecorr

        m1,m2,mask = self.get_m(i)

        g1 = self.mcal_g1[mask]
        g2 = self.mcal_g2[mask]
        n_i = len(g1)
        cat_i = treecorr.Catalog(
            g1 = (g1 - g1.mean()) / m1,
            g2 = (g2 - g2.mean()) / m2,
            ra=self.ra[mask], dec=self.dec[mask],
            ra_units='degree', dec_units='degree')
        del g1, g2

        mask = self.lens_gals == j
        random_mask = self.random_binning==j
        n_j = mask.sum()
        n_rand = random_mask.sum()

        cat_j = treecorr.Catalog(
            ra=self.ra[mask],
            dec=self.dec[mask],
            ra_units='degree',
            dec_units='degree')

        rancat_j  = treecorr.Catalog(
            ra=self.random_ra[random_mask],
            dec=self.random_dec[random_mask],
            ra_units='degree',
            dec_units='degree')

        print(f"Rank {self.rank} calculating shear-position bin pair ({i},{j}): {n_i} x {n_j} objects, {n_rand} randoms")

        ng = treecorr.NGCorrelation(self.config)
        rg = treecorr.NGCorrelation(self.config)

        ng.process(cat_j, cat_i)
        rg.process(rancat_j, cat_i)

        gammat,gammat_im,gammaterr=ng.calculateXi(rg=rg)

        theta=np.exp(ng.meanlogr)
        gammaterr=np.sqrt(gammaterr)

        return theta, gammat, gammaterr, ng.npairs, ng.weight

    def calc_pos_pos(self,i,j):
        import treecorr

        mask = self.lens_gals == i
        random_mask = self.random_binning==i
        n_i = mask.sum()
        n_rand_i = random_mask.sum()

        cat_i = treecorr.Catalog(
            ra=self.ra[mask], dec=self.dec[mask],
            ra_units='degree', dec_units='degree')

        rancat_i  = treecorr.Catalog(
            ra=self.random_ra[random_mask], dec=self.random_dec[random_mask],
            ra_units='degree', dec_units='degree')

        if i==j:
            cat_j = cat_i
            rancat_j = rancat_i
            n_j = n_i
            n_rand_j = n_rand_i
        else:
            mask = self.lens_gals == j
            random_mask = self.random_binning==j
            n_j = mask.sum()
            n_rand_j = random_mask.sum()

            cat_j = treecorr.Catalog(
                ra=self.ra[mask], dec=self.dec[mask],
                ra_units='degree', dec_units='degree')

            rancat_j  = treecorr.Catalog(
                ra=self.random_ra[random_mask], dec=self.random_dec[random_mask],
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

    def load_tomography(self):

        # Columns we need from the tomography catalog
        tom_cols = ['source_bin']
        bias_cols = ['R_gamma'] #TODO R_S - see Sub.Sec. 4.1 in DES Y1 paper R = Rgamma + Rs
        binning = []

        f = self.open_input('tomography_catalog')
        self.binning = f['tomography/source_bin'][:]
        self.lens_gals = f['tomography/lens_bin'][:]
        f.close()

        f = self.open_input('tomography_catalog')
        self.r_gamma = f['multiplicative_bias/R_gamma'][:]
        f.close()

    def load_shear_catalog(self):

        # Columns we need from the shear catalog
        cat_cols = ['ra','dec','mcal_g','mcal_flags',]
        # JAZ I couldn't see a use for these at the moment - will probably need them later,
        # though may be able to do those algorithms on-line
        # cat_cols += ['mcal_mag','mcal_s2n_r', 'mcal_T']

        print(f"Loading shear catalog columns: {cat_cols}")

        f = self.open_input('shear_catalog')
        data = f[1].read_columns(cat_cols)

        mcal_g1 = data['mcal_g'][:,0]
        mcal_g2 = data['mcal_g'][:,1]

        if self.config['flip_g2']:
            mcal_g2 *= -1

        ra = data['ra']
        dec = data['dec']


        self.mcal_g1 = mcal_g1
        self.mcal_g2 = mcal_g2
        self.ra = ra
        self.dec = dec

    def load_random_catalog(self):

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','e1','e2','ra','bin']
        print(f"Loading random catalog columns: {randoms_cols}")

        f = self.open_input('random_cats')
        data = f['randoms']

        cut = self.config['reduce_randoms_size']
        if 0.0<cut<1.0:
            N = data['dec'].size
            sel = np.random.uniform(size=N) < cut
        else:
            sel = slice(None)

        self.random_dec = data['dec'][sel]
        self.random_e1 =  data['e1'][sel]
        self.random_e2 =  data['e2'][sel]
        self.random_ra =  data['ra'][sel]
        self.random_binning = data['bin'][sel]

        f.close()


    def calc_sigma_e(self, nbins_source):
        """
        Calculate sigma_e for shape catalog.
        """
        sigma_e_list = []
        mean_g1_list = []
        mean_g2_list = []
        for i in range(nbins_source):
            m1, m2, mask = self.get_m(i)
            s = (m1+m2)/2
            g1 = self.mcal_g1[mask]
            g2 = self.mcal_g2[mask]
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

    def calc_neff(self, area, nbins_source):
        neff = []
        for i in range(nbins_source):
            m1, m2, mask = self.get_m(i)
            w    = np.ones(len(self.ra[mask]))
            a    = np.sum(w)**2
            b    = np.sum(w**2)
            c    = area
            neff.append(a/b/c)
        return neff

    def calc_area(self):
        import healpy as hp
        pix=hp.ang2pix(4096, np.pi/2.-np.radians(self.dec),np.radians(self.ra), nest=True)
        area=hp.nside2pixarea(4096)*(180./np.pi)**2
        mask=np.bincount(pix)>0
        area=np.sum(mask)*area
        area=float(area) * 60. * 60.
        return area

    def calc_metadata(self, nbins_source):
        #TODO put the metadata in the output SACC file
        import yaml
        area = self.calc_area()
        neff = self.calc_neff(area, nbins_source)
        sigma_e, mean_e1, mean_e2 = self.calc_sigma_e(nbins_source)
        data = {
            "neff": neff,
            "area": area,
            "sigma_e": sigma_e,
            "mean_e1": mean_e1,
            "mean_e2": mean_e2,
        }

        return data

if __name__ == '__main__':
    PipelineStage.main()
