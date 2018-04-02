from pipette import PipelineStage
from descformats.tx import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, PhotozPDFFile
import h5py
import numpy as np
import treecorr

CORES_PER_TASK=20
num_calcs = 1


class TXTwoPoint(PipelineStage):
    name='TXTwoPoint'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog),
        ('config', YamlFile),
    ]
    outputs = [
        ('twopoint_data', TomographyCatalog), #TODO change to a two point SACC file or hdf by another name
    ]
    # TODO Add values to the config file that are not previously defined
    #config_options = {'binslop':None}

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

        #Load the tomography catalog
        self.load_tomography()

        #Load the shear catalog
        self.load_shear_catalog()

        k=0 # Just try shear-shear for now
        all_calcs = [(i,j) for i in range(nbins) for j in range(nbins)]
        calcs=[]

        self.theta = []
        self.xip = []
        self.xim = []
        self.xiperr = []
        self.ximerr = []
        self.npairs = []
        self.weight = []
        self.calc = []

        for i,j in all_calcs:
            #if (k==0)&(i<=j):
            calcs.append((i,j))
            #if self.params['lensfile'] != 'None':
            #    if (k==1)&(i<self.lens_zbins)&(j<self.zbins):
            #        calcs.append((i,j,k))
            #    if (k==2)&(i<=j)&(j<self.lens_zbins):
            #        calcs.append((i,j,k))
        for calc in calcs:
            self.call_treecorr(calc[0],calc[1])

        self.write_output(output_file)
        output_file.close()

    def setup_output(self, info):
        #n = self.open_input('shear_catalog')[1].get_nrows()
        outfile = self.open_output('twopoint_data', parallel=True)
        group = outfile.create_group('twopoint')

        group.create_dataset('bin', (num_calcs,), dtype='i')

        return outfile

    def write_output(self,outfile):
        group = outfile['twopoint']
        group['theta'] = self.theta
        group['xip'] = self.xip
        group['xim'] = self.xim
        group['xiperr'] = self.xiperr
        group['ximerr'] = self.ximerr
        group['npairs'] = self.npairs
        group['weight'] = self.weight
        group['calc'] = self.calc

        #print('theta is', self.theta)
        #print('theta type is', type(self.theta))

        #print('xip is', self.xip)
        #print('xip type is', type(self.xip))

        #print('xim is', self.xim)
        #print('xim type is', type(self.xim))

        #print('xiperr is', self.xiperr)
        #print('xiperr type is', type(self.xiperr))

        #print('ximerr is', self.ximerr)
        #print('ximerr type is', type(self.ximerr))

        #print('npairs is', self.npairs)
        #print('npairs type is', type(self.npairs))

        #print('weight is', self.weight)
        #print('weight type is', type(self.weight))

        #print('calc is', self.calc)
        #print('calc type is', type(self.calc))

    def call_treecorr(self,i,j):
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

        #k=0
        #if (k==0): # xi+-
        theta, xip, xim, xiperr, ximerr, npairs, weight = self.calc_shear_shear(i,j)
            #self.xi.append([xi,xi2,None,None])
            #self.xierr.append([xierr,xi2err,None,None])
        #if (k==1): # gammat
        #    theta,xi,xierr,npairs,weight = self.calc_pos_shear(i,j,verbose,num_threads)
        #    self.xi.append([None,None,xi,None])
        #    self.xierr.append([None,None,xierr,None])
        #if (k==2): # wtheta
        #    theta,xi,xierr,npairs,weight = self.calc_pos_pos(i,j,verbose,num_threads)
        #    self.xi.append([None,None,None,xi])
        #    self.xierr.append([None,None,None,xierr])

        if i==j:
            npairs/=2
            weight/=2
        self.theta.append(theta)
        self.xip.append(xip)
        self.xim.append(xim)
        self.xiperr.append(xiperr)
        self.ximerr.append(ximerr)
        self.npairs.append(npairs)
        self.weight.append(weight)
        self.calc.append((i,j))

    def get_m(self,i):

        mask = (self.binning == i)

        m1 = 0
        m2 = 0

        #m1 = self.shape['m1'] #TODO find m1 and m2 in the shear catalog or elsewhere
        #m2 = self.shape['m2']
        return m1, m2, mask

    def calc_shear_shear(self,i,j):

        # Define the binning.  Binning in TreeCorr uses bins that are equally spaced in log(r).
        # (i.e. Natural log, not log10.)  There are four parameters of which you may specify any 3.
        min_sep= .01          # The minimum separation that you want included.
        max_sep= 3        # The maximum separation that you want included.
        #nbins= 100          # The number of bins
        bin_size= 0.06     # The width of the bins in log(r).  In this case automatically calculated
                    # to be bin_size = log(max_sep/min_sep) / nbins ~= 0.06
        sep_units= 'degree'   # The units of min_sep, max_sep TODO Figure out what we actually want from these- add them to the configuration file maybe?

        m1,m2,mask = self.get_m(i)
        print('The size of cat_i is,', len(self.mcal_g1[mask]))

        print(self.dec)

        #TODO apply results of mcal
        cat_i = treecorr.Catalog(g1 = self.mcal_g1[mask], g2 = self.mcal_g2[mask],ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #bias_cat_i  = treecorr.Catalog(m1,m2) #TODO add bias catalog

        m1,m2,mask = self.get_m(j)

        print('The size of cat_j is,', len(self.mcal_g1[mask]))

        cat_j = treecorr.Catalog(g1=self.mcal_g1[mask], g2 = self.mcal_g2[mask], ra=self.ra[mask], dec=self.dec[mask], ra_units='degree', dec_units='degree')
        #biascat_j = treecorr.Catalog(m1,m2)

        gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, sep_units=sep_units)
        gg.process(cat_i,cat_j)
        print('bin_size = ',gg.bin_size)
        #kk = treecorr.KKCorrelation(nbins=self.params['tbins'], min_sep=self.params['tbounds'][0], max_sep=self.params['tbounds'][1], sep_units='arcmin', bin_slop=self.params['slop'], verbose=verbose,num_threads=num_threads)
        #kk.process(biascat_i,biascat_j)
        #norm = kk.xi

        theta=np.exp(gg.meanlogr)
        xip = gg.xip#/norm
        xim = gg.xim#/norm
        xiperr = ximerr = np.sqrt(gg.varxi)#/norm

        #gg.write('test_twopoint')

        return theta, xip, xim, xiperr, ximerr, gg.npairs, gg.weight

    def load_tomography(self):

        # Columns we need from the tomography catalog
        tom_cols = ['bin']
        bias_cols = ['R_gamma'] #TODO R_S

        chunk_rows = 1084192

        for start, end, data in self.iterate_hdf('tomography_catalog','tomography',tom_cols, chunk_rows):
            self.binning = data['bin']

        for start, end, data in self.iterate_hdf('tomography_catalog','multiplicative_bias',bias_cols, chunk_rows):
            self.r_gamma = data['R_gamma']

    def load_shear_catalog(self):
        # Columns we need from the shear catalog
        #cat_cols = ['mcal_flags', 'mcal_Tpsf']

        # Including all the metacalibration variants of these columns
        #for c in ['mcal_T', 'mcal_s2n_r', 'mcal_g']:
        #    cat_cols += [c, c+"_1p", c+"_1m", c+"_2p", c+"_2m", ]

        # Columns we need from the shear catalog
        cat_cols = ['ra','dec','mcal_g']
        chunk_rows = 1084192 # We are looking at all the data at once for now
        iter_shear = self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows)

        for start, end, data in self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows):
            mcal_g1 = data['mcal_g'][:,0]
            mcal_g2 = data['mcal_g'][:,1]
            ra = data['ra']
            dec = data['dec']

        mask_g1 = (mcal_g1 != -9999.0)
        mask_g2 = (mcal_g2 != -9999.0)

        mcal_g1 = mcal_g1[mask_g1&mask_g2]
        mcal_g2 = mcal_g2[mask_g1&mask_g2]

        ra = ra[mask_g1&mask_g2]
        dec = dec[mask_g1&mask_g2]

        self.mcal_g1 = mcal_g1
        self.mcal_g2 = mcal_g2
        self.ra = ra
        self.dec = dec
        self.binning = self.binning[mask_g1&mask_g2]



if __name__ == '__main__':
    PipelineStage.main()
