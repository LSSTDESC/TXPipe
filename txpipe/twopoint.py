from ceci import PipelineStage
from descformats.tx import HDFFile, MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, PhotozPDFFile
import h5py
import numpy as np
import treecorr
import random



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

        import os

        output_file = self.setup_output()

        # MPI Information
        rank = self.rank
        size = self.size
        comm = self.comm

        if comm:
            comm.Barrier()

        # Read in the number of bins

        zbins = self.read_zbins()
        nbins = len(zbins)

        print(f'Running with {nbins} tomographic bins')

        self.load_tomography()

        self.load_shear_catalog()

        self.load_random_catalog()

        calcs = []
        for k in [0,1,2]:
            for i in range(nbins):
                for j in range(nbins):
                    # type of correlation: 0=shear-shear, 1=shear-pos, 2=pos-pos
                    # For shear-shear and pos-pos we don't need both
                    # i,j and j,i (by symmetry they are the same)
                    if (j>i) and k in [0,2]:
                        continue
                    else:
                        calcs.append((i,j,k))


        self.setup_functions()

        for i,j,k in calcs:
            self.call_treecorr(i, j, k)

        self.write_output(output_file)
        output_file.close()

    def setup_functions(self):

        self.theta_gg = []
        self.xip = []
        self.xim = []
        self.xiperr = []
        self.ximerr = []
        self.npairs_gg = []
        self.weight_gg = []
        self.bin_ij_gg = []

        self.theta_ng = []
        self.gammat = []
        self.gammaterr = []
        self.npairs_ng = []
        self.weight_ng = []
        self.bin_ij_ng = []

        self.theta_nn = []
        self.wtheta = []
        self.wthetaerr = []
        self.npairs_nn = []
        self.weight_nn = []
        self.bin_ij_nn = []



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
        group['theta_gg'] = self.theta_gg
        group['xip'] = self.xip
        group['xim'] = self.xim
        group['xiperr'] = self.xiperr
        group['ximerr'] = self.ximerr
        group['npairs_gg'] = self.npairs_gg
        group['weight_gg'] = self.weight_gg
        group['bin_ij_gg'] = self.bin_ij_gg

        group['theta_ng'] = self.theta_ng
        group['gammat'] = self.gammat
        group['gammaterr'] = self.gammaterr
        group['npairs_ng'] = self.npairs_ng
        group['weight_ng'] = self.weight_ng
        group['bin_ij_ng'] = self.bin_ij_ng

        group['theta_nn'] = self.theta_nn
        group['wtheta'] = self.wtheta
        group['wthetaerr'] = self.wthetaerr
        group['npairs_nn'] = self.npairs_nn
        group['weight_nn'] = self.weight_nn
        group['bin_ij_nn'] = self.bin_ij_nn

    def call_treecorr(self,i,j,k):
        """
        This is a wrapper for interaction with treecorr.
        """
        # k==0: xi+-
        # k==1: gammat
        # k==2: wtheta

        # Cori value

        if (k==0): # xi+-
            theta_gg,xip, xim, xiperr, ximerr, npairs_gg, weight_gg = self.calc_shear_shear(i,j)
            if i==j:
                npairs_gg/=2
                weight_gg/=2
            self.theta_gg.append(theta_gg)
            self.xip.append(xip)
            self.xim.append(xim)
            self.xiperr.append(xiperr)
            self.ximerr.append(ximerr)
            self.npairs_gg.append(npairs_gg)
            self.weight_gg.append(weight_gg)
            self.bin_ij_gg.append((i,j))

        if (k==1): # gammat
            theta_ng, gammat, gammaterr, npairs_ng, weight_ng = self.calc_pos_shear(i,j)
            if i==j:
                npairs_ng/=2
                weight_ng/=2
            self.theta_ng.append(theta_ng)
            self.gammat.append(gammat)
            self.gammaterr.append(gammaterr)
            self.npairs_ng.append(npairs_ng)
            self.weight_ng.append(weight_ng)
            self.bin_ij_ng.append((i,j))

        if (k==2): # wtheta
            theta_nn,wtheta,wthetaerr,npairs_nn,weight_nn = self.calc_pos_pos(i,j)
            if i==j:
                npairs_nn/=2
                weight_nn/=2
            self.theta_nn.append(theta_nn)
            self.wtheta.append(wtheta)
            self.wthetaerr.append(wthetaerr)
            self.npairs_nn.append(npairs_nn)
            self.weight_nn.append(weight_nn)
            self.bin_ij_nn.append((i,j))



    def get_m(self,i):

        mask = (self.binning == i)

        #[:1066016]
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

        cat_j = treecorr.Catalog(g1 = (self.mcal_g1[mask] - np.mean(self.mcal_g1[mask]))/m1, g2 = (self.mcal_g2[mask] - np.mean(self.mcal_g2[mask]))/m2 ,ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')

        gg = treecorr.GGCorrelation(self.config)
        gg.process(cat_i,cat_j)

        theta=np.exp(gg.meanlogr)
        xip = gg.xip
        xim = gg.xim
        xiperr = ximerr = np.sqrt(gg.varxi)

        #gg.write('test_twopoint')
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
            mcal_g2 = -mcal_g2
        # mcal_mag = data['mcal_mag']
        # mcal_s2n = data['mcal_s2n_r']
        # mcal_T = data['mcal_T']
        ra = data['ra']
        dec = data['dec']
        flags = data['mcal_flags']
        #weights = data['mcal_weight']

        cut1  = (flags == 0)
        #cut2 = (data['mcal_s2n_r'] > 10)
        #cut3 = (data['mcal_T'] / data['psfrec_T']) > 0.5)

        mask = cut1#&cut2&cut3

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

        #mag_cut = self.mcal_mag < 21
        bin_cut = self.binning == 0

        mask = bin_cut#&mag_cut

        m1 = np.mean(self.r_gamma[mask][:,0,0]) # R11, taking the mean for the bin
        m2 = np.mean(self.r_gamma[mask][:,1,1]) #R22

        return m1, m2, mask



if __name__ == '__main__':
    PipelineStage.main()
