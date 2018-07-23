from ceci import PipelineStage
from descformats.tx import HDFFile, MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, PhotozPDFFile
import h5py
import numpy as np
import treecorr
import random
import collections
import sys

# This creates a little mini-type, like a struct,
# for holding individual measurements
Measurement = collections.namedtuple(
    'Measurement', 
    ['corr_type', 'theta', 'value', 'error', 'npair', 'weight', 'i', 'j'])


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('random_cats', RandomsCatalog)
    ]
    outputs = [
        ('twopoint_data', HDFFile), #TODO possibly change to a SACC file, change to TwoPoint once class added to DESCFormats
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
        }

    def run(self):
        """
        The run method is where all the work of the stage is done.
        In this case it:
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
        zbins = self.read_zbins()
        nbins = len(zbins)

        print(f'Running with {nbins} tomographic bins')

        # Various setup and input functions.
        self.load_tomography()
        self.load_shear_catalog()
        self.load_random_catalog()
        self.setup_results()

        calcs = self.select_calculations(nbins)
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
            output_file = self.setup_output()
            self.write_output(output_file)
            output_file.close()


    def select_calculations(self, nbins):
        calcs = []
        # type of correlation: 0=shear-shear, 1=shear-pos, 2=pos-pos
        for k in [0,1,2]:
            for i in range(nbins):
                for j in range(nbins):
                    # For shear-shear and pos-pos we don't need both
                    # i,j and j,i (by symmetry they are the same)
                    if (j>i) and k in [0,2]:
                        continue
                    else:
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


    def read_zbins(self):
        tomo = self.open_input('tomography_catalog')
        d = dict(tomo['tomography'].attrs)
        tomo.close()
        nbin = d['nbin']
        zbins = [(d[f'zmin_{i}'], d[f'zmax_{i}']) for i in range(nbin)]
        return zbins

    def setup_output(self):
        outfile = self.open_output('twopoint_data')
        group = outfile.create_group('twopoint')
        return outfile

    def write_output(self,outfile):
        # TODO fix this to account for the case where we only do a certain number of calcs
        group = outfile['twopoint']
        fields = ['corr_type', 'theta', 'value', 'error', 'npair', 'weight', 'i', 'j']
        output = {f:list() for f in fields}

        for corr_type in [b'xip', b'xim', b'gammat', b'wtheta']:
            data = [r for r in self.results if r.corr_type==corr_type]
            for bin_pair_data in data:
                n = len(bin_pair_data.theta)
                for f in fields:
                    v = getattr(bin_pair_data, f)
                    if np.isscalar(v):
                        v = [v for i in range(n)]
                    else:
                        v = v.tolist()
                    output[f] += v

        for name, values in output.items():
            values = np.array(values)
            group[name] = values

    def call_treecorr(self,i,j,k):
        """
        This is a wrapper for interaction with treecorr.
        """
        # k==0: xi+-
        # k==1: gammat
        # k==2: wtheta

        # Cori value


        if (k==0): # xi+-
            theta, xip, xim, xiperr, ximerr, npairs, weight = self.calc_shear_shear(i,j)
            if i==j:
                npairs/=2
                weight/=2

            self.results.append(Measurement(b"xip", theta, xip, xiperr, npairs, weight, i, j))
            self.results.append(Measurement(b"xim", theta, xim, ximerr, npairs, weight, i, j))

        elif (k==1): # gammat
            theta, val, err, npairs, weight = self.calc_pos_shear(i,j)
            if i==j:
                npairs/=2
                weight/=2

            self.results.append(Measurement(b"gammat", theta, val, err, npairs, weight, i, j))

        elif (k==2): # wtheta
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
        print(f"Calculating shear-shear bin pair ({i},{j})")
        m1,m2,mask = self.get_m(i)

        cat_i = treecorr.Catalog(
            g1 = (self.mcal_g1[mask] - np.mean(self.mcal_g1[mask]))/m1, 
            g2 = (self.mcal_g2[mask] - np.mean(self.mcal_g2[mask]))/m2,
            ra=self.ra[mask], dec=self.dec[mask], 
            ra_units='degree', dec_units='degree')

        m1,m2,mask = self.get_m(j)

        cat_j = treecorr.Catalog(
            g1 = (self.mcal_g1[mask] - np.mean(self.mcal_g1[mask]))/m1,
            g2 = (self.mcal_g2[mask] - np.mean(self.mcal_g2[mask]))/m2,
            ra=self.ra[mask], dec=self.dec[mask],
            ra_units='degree', dec_units='degree')

        gg = treecorr.GGCorrelation(self.config)
        gg.process(cat_i,cat_j)

        theta=np.exp(gg.meanlogr)
        xip = gg.xip
        xim = gg.xim
        xiperr = ximerr = np.sqrt(gg.varxi)

        return theta, xip, xim, xiperr, ximerr, gg.npairs, gg.weight

    def calc_pos_shear(self,i,j):
        print(f"Calculating position-shear bin pair ({i},{j})")

        #TODO check if we want to subtract out a mean shear
        m1, m2, lensmask = self.select_lens()

        ranmask_i = self.random_binning==i

        cat_lens = treecorr.Catalog(
            ra=self.ra[lensmask], 
            dec=self.dec[lensmask], 
            ra_units='degree',
            dec_units='degree')

        m1,m2,mask = self.get_m(j)

        cat_j = treecorr.Catalog(
            g1 = (self.mcal_g1[mask] - np.mean(self.mcal_g1[mask]))/m1, 
            g2 = (self.mcal_g2[mask] - np.mean(self.mcal_g2[mask]))/m2,
            ra=self.ra[mask], 
            dec=self.dec[mask], 
            ra_units='degree', 
            dec_units='degree')

        mask = self.random_binning==i

        rancat_i  = treecorr.Catalog(
            ra=self.random_ra[ranmask_i], 
            dec=self.random_dec[ranmask_i], 
            ra_units='degree', 
            dec_units='degree')

        ng = treecorr.NGCorrelation(self.config)
        rg = treecorr.NGCorrelation(self.config)

        ng.process(cat_lens,cat_j)
        rg.process(rancat_i,cat_j)

        gammat,gammat_im,gammaterr=ng.calculateXi(rg)

        theta=np.exp(ng.meanlogr)
        gammaterr=np.sqrt(gammaterr)

        return theta, gammat, gammaterr, ng.npairs, ng.weight

    def calc_pos_pos(self,i,j):
        print(f"Calculating position-position bin pair ({i},{j})")

        m1, m2, lensmask = self.select_lens()

        cat_lens = treecorr.Catalog(
            ra=self.ra[lensmask], dec=self.dec[lensmask],
            ra_units='degree', dec_units='degree')

        m1,m2,mask = self.get_m(j)

        cat_j = treecorr.Catalog(
            ra=self.ra[mask], dec=self.dec[mask],
            ra_units='degree', dec_units='degree')

        mask = self.random_binning==i
        rancat_i  = treecorr.Catalog(
            ra=self.random_ra[mask], dec=self.random_dec[mask],
            ra_units='degree', dec_units='degree')

        mask = self.random_binning==j
        rancat_j  = treecorr.Catalog(
            ra=self.random_ra[mask], dec=self.random_dec[mask],
            ra_units='degree', dec_units='degree')


        nn = treecorr.NNCorrelation(self.config)
        rn = treecorr.NNCorrelation(self.config)
        nr = treecorr.NNCorrelation(self.config)
        rr = treecorr.NNCorrelation(self.config)

        nn.process(cat_lens)
        rn.process(rancat_i, cat_lens)
        nr.process(cat_lens, rancat_j)
        rr.process(rancat_i, rancat_j)

        theta=np.exp(nn.meanlogr)
        wtheta,wthetaerr=nn.calculateXi(rr,dr=nr,rd=rn)
        wthetaerr=np.sqrt(wthetaerr)

        return theta, wtheta, wthetaerr, nn.npairs, nn.weight

    def load_tomography(self):

        # Columns we need from the tomography catalog
        tom_cols = ['bin']
        bias_cols = ['R_gamma'] #TODO R_S - see Sub.Sec. 4.1 in DES Y1 paper R = Rgamma + Rs
        binning = []

        f = self.open_input('tomography_catalog')
        self.binning = f['tomography/bin'][:]
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
        flags = data['mcal_flags']
        # TODO: WEIGHTS

        mask  = (flags == 0)

        self.mcal_g1 = mcal_g1[mask]
        self.mcal_g2 = mcal_g2[mask]
        # self.mcal_mag = mcal_mag[mask]
        # self.mcal_s2n = mcal_s2n[mask]
        # self.mcal_T = mcal_T[mask]
        self.ra = ra[mask]
        self.dec = dec[mask]
        print('len binning', len(self.binning))
        print('len mask', len(mask))
        print('len mcal g1 ', len(mcal_g1))
        self.binning = self.binning[mask]

    def load_random_catalog(self):

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','e1','e2','ra','bin']
        print(f"Loading random catalog columns: {randoms_cols}")

        f = self.open_input('random_cats')
        data = f['randoms']
        self.random_dec = data['dec'][:]
        self.random_e1 =  data['e1'][:]
        self.random_e2 =  data['e2'][:]
        self.random_ra =  data['ra'][:]
        self.random_binning = data['bin'][:]
        f.close()


    def select_lens(self):
        # Extremely simple lens selector simply by bin
        mask = self.binning == 0

        m1 = np.mean(self.r_gamma[mask][:,0,0]) # R11, taking the mean for the bin
        m2 = np.mean(self.r_gamma[mask][:,1,1]) #R22

        return m1, m2, mask



if __name__ == '__main__':
    PipelineStage.main()
