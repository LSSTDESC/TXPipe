from ceci import PipelineStage
from descformats.tx import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, PhotozPDFFile
import h5py
import numpy as np
import treecorr
import random

CORES_PER_TASK=20
num_calcs = 1


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('random_cats', RandomsCatalog),
        ('config', YamlFile),
    ]
    outputs = [
        ('twopoint_data', TomographyCatalog), #TODO change to a two point SACC file or hdf by another name
    ]
    # TODO Add values to the config file that are not previously defined
    config_options = {'binslop':None, 'calcs':[0]}

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

        info = self.read_config()
        output_file = self.setup_output(info)

        # Read in the number of bins

        zbin_edges = info['zbin_edges']
        zbins = list(zip(zbin_edges[:-1], zbin_edges[1:]))
        nbins = len(zbins)

        print('number of bins', nbins)
        #Load the tomography catalog
        self.load_tomography()

        #Load the shear catalog
        self.load_shear_catalog()

        #Load the random catalog
        self.load_random_catalog()

        #all_bins = [(i,j) for i in range(nbins) for j in range(nbins)]
        all_bins = [(i,j) for i in np.arange(-1,nbins,1) for j in np.arange(-1,nbins,1)]
        calcs=[]

        for i,j in all_bins:
            calcs.append((i,j))

        for k in [0,1,2]:
            self.setup_functions(k)
            for calc in calcs:
                self.call_treecorr(calc[0], calc[1], k)

        self.write_output(output_file)
        output_file.close()

    def setup_functions(self,k):
        self.calc = []
        print('k is', k)
        if (k==0): #xi+-
            self.theta_gg = []
            self.xip = []
            self.xim = []
            self.xiperr = []
            self.ximerr = []
            self.npairs_gg = []
            self.weight_gg = []
        if (k==1): #gammat
            self.theta_ng = []
            self.gammat = []
            self.gammaterr = []
            self.npairs_ng = []
            self.weight_ng = []
        if (k==2): #wtheta
            self.theta_nn = []
            self.wtheta = []
            self.wthetaerr = []
            self.npairs_nn = []
            self.weight_nn = []

    def setup_output(self, info):
        #n = self.open_input('shear_catalog')[1].get_nrows()
        outfile = self.open_output('twopoint_data', parallel=True)
        group = outfile.create_group('twopoint')

        group.create_dataset('bin', (num_calcs,), dtype='i')

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
        group['calc'] = self.calc

        group['theta_ng'] = self.theta_ng
        group['gammat'] = self.gammat
        group['gammaterr'] = self.gammaterr
        group['npairs_ng'] = self.npairs_ng
        group['weight_ng'] = self.weight_ng

        group['theta_nn'] = self.theta_nn
        group['wtheta'] = self.wtheta
        group['wthetaerr'] = self.wthetaerr
        group['npairs_nn'] = self.npairs_nn
        group['weight_nn'] = self.weight_nn

        #print('theta is', self.theta)
        #print('theta type is', type(self.theta))

    def call_treecorr(self,i,j,k):
        """
        This is a wrapper for interaction with treecorr.
        """
        print("Running 2pt analysis on pair {},{}".format(i, j))
        # k==0: xi+-
        # k==1: gammat
        # k==2: wtheta

        #TODO define these quantities
        #verbose=0
        # Cori value
        #num_threads=CORES_PER_TASK

        # if k!=1:
        #     return 0

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
            self.calc.append((i,j))
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
            self.calc.append((i,j))
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
            self.calc.append((i,j))



    def get_m(self,i):

        mask = (self.binning == i)

        m1 = np.mean(self.r_gamma[:,0,0]) # R11, taking the mean for the bin, TODO check if that's what we want to do
        m2 = np.mean(self.r_gamma[:,1,1]) #R22

        return m1, m2, mask

    def calc_shear_shear(self,i,j):

        #TODO check if we want to subtract out a mean shear

        # Define the binning.  Binning in TreeCorr uses bins that are equally spaced in log(r).
        # (i.e. Natural log, not log10.)  There are four parameters of which you may specify any 3.
        min_sep= 1          # The minimum separation that you want included.
        max_sep= 100        # The maximum separation that you want included.
        #nbins= 100          # The number of bins
        bin_size= 0.06     # The width of the bins in log(r).  In this case automatically calculated
                    # to be bin_size = log(max_sep/min_sep) / nbins ~= 0.06
        sep_units= 'arcmin'   # The units of min_sep, max_sep TODO Figure out what we actually want from these- add them to the configuration file maybe?

        m1,m2,mask = self.get_m(i)
        print('The size of cat_i is,', len(self.mcal_g1[mask]))

        #mask = [bool(random.getrandbits(1)) for i in range(390935)]

        cat_i = treecorr.Catalog(g1 = self.mcal_g1[mask]/m1, g2 = self.mcal_g2[mask]/m2,ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #cat_i = treecorr.Catalog(g1 = self.mcal_g1, g2 = self.mcal_g2,ra=self.ra, dec=self.dec, ra_units='degree', dec_units='degree')

        m1,m2,mask = self.get_m(j)

        cat_j = treecorr.Catalog(g1=self.mcal_g1[mask]/m1, g2 = -self.mcal_g2[mask]/m2, ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #cat_j = treecorr.Catalog(g1 = self.mcal_g1, g2 = self.mcal_g2,ra=self.ra, dec=self.dec, ra_units='degree', dec_units='degree')

        gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)
        gg.process(cat_i,cat_j)
        print('bin_size = ',gg.bin_size)

        theta=np.exp(gg.meanlogr)
        xip = gg.xip
        xim = gg.xim
        xiperr = ximerr = np.sqrt(gg.varxi)

        #gg.write('test_twopoint')

        # theta_gg,xi,xi2,xierr,xi2err,npairs_gg,weight_gg
        return theta, xip, xim, xiperr, ximerr, gg.npairs, gg.weight

    def calc_pos_shear(self,i,j):
        #TODO figure out how you distinguish between lens and sources

        #TODO check if we want to subtract out a mean shear

        # Define the binning.  Binning in TreeCorr uses bins that are equally spaced in log(r).
        # (i.e. Natural log, not log10.)  There are four parameters of which you may specify any 3.
        min_sep= 1          # The minimum separation that you want included.
        max_sep= 400        # The maximum separation that you want included.
        #nbins= 100          # The number of bins
        bin_size= 0.06     # The width of the bins in log(r).  In this case automatically calculated
                    # to be bin_size = log(max_sep/min_sep) / nbins ~= 0.06
        sep_units= 'arcmin'   # The units of min_sep, max_sep TODO Figure out what we actually want from these- add them to the configuration file maybe?

        m1,m2,mask = self.get_m(i)

        #mask = [bool(random.getrandbits(1)) for i in range(390935)]

        cat_i = treecorr.Catalog(g1 = self.mcal_g1[mask]/m1, g2 = self.mcal_g2[mask]/m2,ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #cat_i = treecorr.Catalog(g1 = self.mcal_g1, g2 = self.mcal_g2,ra=self.ra, dec=self.dec, ra_units='degree', dec_units='degree')
        m1,m2,mask = self.get_m(j)

        cat_j = treecorr.Catalog(g1=self.mcal_g1[mask]/m1, g2 = -self.mcal_g2[mask]/m2, ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #cat_j = treecorr.Catalog(g1 = self.mcal_g1, g2 = self.mcal_g2,ra=self.ra, dec=self.dec, ra_units='degree', dec_units='degree')

        mask = self.random_binning==i
        rancat_i  = treecorr.Catalog(g1 = self.random_e1[mask], g2 = self.random_e2[mask], ra=self.random_ra[mask], dec=self.random_dec[mask], ra_units='deg', dec_units='deg')
        #rancat_i  = treecorr.Catalog(g1 = self.random_e1, g2 = self.random_e2, ra=self.random_ra, dec=self.random_dec, ra_units='deg', dec_units='deg')

        ng = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)
        rg = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)

        ng.process(cat_i,cat_j)
        rg.process(rancat_i,cat_j)

        gammat,gammat_im,gammaterr=ng.calculateXi(rg)

        theta=np.exp(ng.meanlogr)
        gammaterr=np.sqrt(gammaterr)

        # theta_ng, gammat, gammaterr, npairs_ng, weight_ng
        return theta, gammat, gammaterr, ng.npairs, ng.weight

    def calc_pos_pos(self,i,j):
        #TODO check if we want to subtract out a mean shear

        # Define the binning.  Binning in TreeCorr uses bins that are equally spaced in log(r).
        # (i.e. Natural log, not log10.)  There are four parameters of which you may specify any 3.
        min_sep= 1          # The minimum separation that you want included.
        max_sep= 100        # The maximum separation that you want included.
        #nbins= 100          # The number of bins
        bin_size= 0.06     # The width of the bins in log(r).  In this case automatically calculated
                    # to be bin_size = log(max_sep/min_sep) / nbins ~= 0.06
        sep_units= 'arcmin'   # The units of min_sep, max_sep TODO Figure out what we actually want from these- add them to the configuration file maybe?

        m1,m2,mask = self.get_m(i)

        #mask = [bool(random.getrandbits(1)) for i in range(390935)]

        cat_i = treecorr.Catalog(g1 = self.mcal_g1[mask]/m1, g2 = self.mcal_g2[mask]/m2,ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #cat_i = treecorr.Catalog(g1 = self.mcal_g1, g2 = self.mcal_g2,ra=self.ra, dec=self.dec, ra_units='degree', dec_units='degree')

        m1,m2,mask = self.get_m(j)

        cat_j = treecorr.Catalog(g1=self.mcal_g1[mask]/m1, g2 = -self.mcal_g2[mask]/m2, ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #cat_j = treecorr.Catalog(g1 = self.mcal_g1, g2 = self.mcal_g2,ra=self.ra, dec=self.dec, ra_units='degree', dec_units='degree')

        mask = self.random_binning==i
        rancat_i  = treecorr.Catalog(g1 = self.random_e1[mask], g2 = self.random_e2[mask], ra=self.random_ra[mask], dec=self.random_dec[mask], ra_units='deg', dec_units='deg')
        #rancat_i  = treecorr.Catalog(g1 = self.random_e1, g2 = self.random_e2, ra=self.random_ra, dec=self.random_dec, ra_units='deg', dec_units='deg')

        mask = self.random_binning==j
        rancat_j  = treecorr.Catalog(g1 = self.random_e1[mask], g2 = self.random_e2[mask], ra=self.random_ra[mask], dec=self.random_dec[mask], ra_units='deg', dec_units='deg')

        #rancat_j  = treecorr.Catalog(g1 = self.random_e1, g2 = self.random_e2, ra=self.random_ra, dec=self.random_dec, ra_units='deg', dec_units='deg')

        nn = treecorr.NNCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)
        rn = treecorr.NNCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)
        nr = treecorr.NNCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)
        rr = treecorr.NNCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)
        nn.process(cat_i,cat_j)
        rn.process(rancat_i,cat_j)
        nr.process(cat_i,rancat_j)
        rr.process(rancat_i,rancat_j)

        theta=np.exp(nn.meanlogr)
        wtheta,wthetaerr=nn.calculateXi(rr,dr=nr,rd=rn)
        wthetaerr=np.sqrt(wthetaerr)

        #theta_nn,wtheta,wthetaerr,npairs_nn,weight_nn
        return theta, wtheta, wthetaerr, nn.npairs, nn.weight

    def load_tomography(self):

        # Columns we need from the tomography catalog
        tom_cols = ['bin']
        bias_cols = ['R_gamma'] #TODO R_S - see Sub.Sec. 4.1 in DES Y1 paper R = Rgamma + Rs

        chunk_rows = 1084192 #9211556

        for start, end, data in self.iterate_hdf('tomography_catalog','tomography',tom_cols, chunk_rows):
            print('reading in the tomography catalog')
            self.binning = data['bin']

        for start, end, data in self.iterate_hdf('tomography_catalog','multiplicative_bias',bias_cols, chunk_rows):
            self.r_gamma = data['R_gamma']

    def load_shear_catalog(self):

        # Columns we need from the shear catalog
        cat_cols = ['ra','dec','mcal_g','mcal_flags']
        #cat_cols = ['RA','DEC','GAMMA1','GAMMA2']
        #cat_cols = ['ra', 'dec','shear_1','shear_2']
        chunk_rows = 1084192 #9211556 # We are looking at all the data at once for now
        iter_shear = self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows)

        for start, end, data in self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows):

            #mcal_g1 = data['GAMMA1']
            #mcal_g2 = data['GAMMA2']
            #ra = data['RA']
            #dec = data['DEC']
            mcal_g1 = data['mcal_g'][:,0]
            mcal_g2 = data['mcal_g'][:,1]
            #mcal_g1 = data['shear_1']
            #mcal_g2 = data['shear_2']
            ra = data['ra']
            dec = data['dec']
            flags = data['mcal_flags']
            #weights = data['mcal_weight']

            mask = (flags == 0)

        self.mcal_g1 = mcal_g1[mask]
        self.mcal_g2 = mcal_g2[mask]
        self.ra = ra[mask]
        self.dec = dec[mask]
        #self.weights = weights[mask]
        self.binning = self.binning[mask]

    def load_random_catalog(self):

        # Columns we need from the tomography catalog
        randoms_cols = ['dec','e1','e2','ra','bin']

        chunk_rows = 1066016 #9211556

        for start, end, data in self.iterate_hdf('random_cats','randoms',randoms_cols, chunk_rows):
            self.random_dec = data['dec']
            self.random_e1 = data['e1']
            self.random_e2 = data['e2']
            self.random_ra = data['ra']
            self.random_binning = data['bin']
            print('randoms binning', self.random_binning)



if __name__ == '__main__':
    PipelineStage.main()
