from ceci import PipelineStage
from descformats.tx import DiagnosticMaps, YamlFile, RandomsCatalog, MetacalCatalog, TomographyCatalog
import numpy as np


class TXRandomCat(PipelineStage):
    name='TXRandomCat'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('tomography_catalog', TomographyCatalog)
    ]
    outputs = [
        ('random_cats', RandomsCatalog),
    ]
    config_options = {
        'density': 100.,  # number per square arcmin at median depth depth.  Not sure if this is right.
        'chunk_rows': 9211556,  # number per square arcmin at median depth depth.  Not sure if this is right.
        'Mstar': 23.0,  # Schecther distribution Mstar parameter
        'alpha': -1.25,  # Schecther distribution Mstar parameter
        'sigma_e': 0.27,
    }

    def run(self):
        import scipy.special
        import scipy.stats
        import healpy
        from . import randoms

        config = self.config

        self.load_tomography(config)
        self.load_shear_catalog(config)
        self.randomize()

        output_file = self.open_output('random_cats')
        group = output_file.create_group('randoms')
        group = output_file['randoms']

        print('ra',len(self.ra_randoms))
        print('dec',len(self.dec_randoms))
        print('g1',len(self.g1_randoms))
        print('g2',len(self.g2_randoms))
        print('bins',len(self.binnings_randoms))

        group['ra'] = self.ra_randoms
        group['dec'] = self.dec_randoms
        group['e1'] = self.g1_randoms
        group['e2'] = self.g2_randoms
        group['bin'] = self.binnings_randoms

        output_file.close()

    def load_shear_catalog(self,config):

        # Columns we need from the shear catalog
        cat_cols = ['ra','dec','mcal_g','mcal_flags']
        #cat_cols = ['RA','DEC','GAMMA1','GAMMA2']
        #cat_cols = ['ra','dec','shear_1','shear_2']
        chunk_rows = config['chunk_rows'] # We are looking at all the data at once for now
        #chunk_rows = 9211556
        iter_shear = self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows)

        for start, end, data in self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows):

            #mcal_g1 = data['GAMMA1']
            #mcal_g2 = data['GAMMA2']
            #ra = data['RA']
            #dec = data['DEC']
            #mcal_g1 = data['shear_1']
            #mcal_g2 = data['shear_2']
            mcal_g1 = data['mcal_g'][:,0]
            mcal_g2 = data['mcal_g'][:,1]
            ra = data['ra']
            dec = data['dec']
            flags = data['mcal_flags']
            #weights = data['mcal_weight']

        #mask = (flags == 0)

        self.mcal_g1 = mcal_g1#[mask]
        self.mcal_g2 = mcal_g2#[mask]
        self.ra = ra#[mask]
        self.dec = dec#[mask]
        #self.weights = weights[mask]


        print('bins are', self.binning)

    def load_tomography(self,config):

        # Columns we need from the tomography catalog
        tom_cols = ['bin']
        bias_cols = ['R_gamma'] #TODO R_S - see Sub.Sec. 4.1 in DES Y1 paper R = Rgamma + Rs

        chunk_rows = config['chunk_rows']

        for start, end, data in self.iterate_hdf('tomography_catalog','tomography',tom_cols, chunk_rows):
            self.binning = data['bin']

        for start, end, data in self.iterate_hdf('tomography_catalog','multiplicative_bias',bias_cols, chunk_rows):
            self.r_gamma = data['R_gamma']

    def randomize(self):
        # Create a simple random catalog

        ra_randoms = []
        dec_randoms = []
        g1_randoms = []
        g2_randoms = []
        binnings_randoms = []

        for bin in [-1,  0,  1,  2,  3]:
            mask = (self.binning == bin)

            ra_min = np.min(self.ra[mask])
            ra_max = np.max(self.ra[mask])
            dec_min = np.min(self.dec[mask])
            dec_max = np.max(self.dec[mask])
            g1_min = np.min(self.mcal_g1[mask])
            g1_max =  np.max(self.mcal_g1[mask])
            g2_max = np.max(self.mcal_g2[mask])
            g2_min = np.min(self.mcal_g2[mask])
            size = len(self.mcal_g2[mask])
            n = 1

            #rand_ra = np.random.uniform(ra_min, ra_max, len(self.ra[mask])).tolist()
            #rand_sindec = np.random.uniform(np.sin(dec_min), np.sin(dec_max), len(self.ra[mask])).tolist()
            #rand_dec = np.arcsin(rand_sindec).tolist()
            
            rand_ra = np.random.uniform(ra_min, ra_max, size=n*size).tolist()
            dec = np.random.uniform(np.sin(np.deg2rad(dec_min)), np.sin(np.deg2rad(dec_max)), size=n*size)
            dec = np.arcsin(dec, out=dec)
            rand_dec = np.rad2deg(dec, out=dec).tolist()

            rand_g1 = np.random.uniform(g1_min, g1_max, len(self.mcal_g1[mask])).tolist()
            rand_g2 = np.random.uniform(g2_min, g2_max, len(self.mcal_g2[mask])).tolist()

            ra_randoms += rand_ra
            dec_randoms += rand_dec
            g1_randoms += rand_g1
            g2_randoms += rand_g2

            binnings_randoms += [bin]*len(self.ra[mask])*n

        self.ra_randoms = ra_randoms
        self.dec_randoms = dec_randoms
        self.g1_randoms = g1_randoms
        self.g2_randoms = g2_randoms
        self.binnings_randoms = binnings_randoms


if __name__ == '__main__':
    PipelineStage.main()
