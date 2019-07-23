"""
People sometimes think that these statistics are called the Rho statistics,
because we usually use that letter for them.  Not so.  They are named after
the wonderfully incorrigible rogue Barney Rowe, now sadly lost to high finance,
who presented the first two of them in MNRAS 404, 350 (2010).
"""

from .base_stage import PipelineStage
from .data_types import PNGFile, HDFFile
import numpy as np

class TXRoweStatistics(PipelineStage):
    name = 'TXRoweStatistics'

    inputs =[('star_catalog', HDFFile)]
    outputs = [
        ('rowe134', PNGFile),
        ('rowe25', PNGFile),
        ('rowe_stats', HDFFile),
    ]

    config_options = {
        'min_sep':0.5,
        'max_sep':250.0,
        'nbins':20,
        'bin_slop':0.01,
        'sep_units':'arcmin',        
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib
        matplotlib.use('agg')

        ra, dec, e_psf, de_psf, T_frac = self.load_stars()

        rowe_stats = {}
        rowe_stats[1] = self.compute_statistic(1, ra, dec, de_psf,       de_psf      )
        rowe_stats[2] = self.compute_statistic(2, ra, dec, e_psf,        de_psf      )
        rowe_stats[3] = self.compute_statistic(3, ra, dec, e_psf*T_frac, e_psf*T_frac)
        rowe_stats[4] = self.compute_statistic(4, ra, dec, de_psf,       e_psf*T_frac)
        rowe_stats[5] = self.compute_statistic(5, ra, dec, e_psf,        e_psf*T_frac)

        self.save_stats(rowe_stats)
        self.rowe_plots(rowe_stats)


    
    def load_stars(self):
        f = self.open_input('star_catalog')
        g = f['stars']
        ra = g['ra'][:]
        dec = g['dec'][:]
        e1 = g['measured_e1'][:]
        e2 = g['measured_e2'][:]
        de1 = e1 - g['model_e1'][:]
        de2 = e2 - g['model_e2'][:]
        T_frac = (g['measured_T'][:] - g['model_T'][:]) / g['measured_T'][:]

        e_psf = np.array((e1, e2))
        de_psf = np.array((de1, de2))

        return ra, dec, e_psf, de_psf, T_frac

    def compute_statistic(self, i, ra, dec, q1, q2):
        n = len(ra)
        print(f"Computing Rowe statistic rho_{i} from {n} objects")
        import treecorr
        corr = treecorr.GGCorrelation(self.config)
        cat1 = treecorr.Catalog(ra=ra, dec=dec, g1=q1[0], g2=q1[1], ra_units='deg', dec_units='deg')
        cat2 = treecorr.Catalog(ra=ra, dec=dec, g1=q2[0], g2=q2[1], ra_units='deg', dec_units='deg')
        corr.process(cat1, cat2)
        return corr.meanr, corr.xip, corr.varxip**0.5

    def rowe_plots(self, rowe_stats):
        # First plot - stats 1,3,4
        import matplotlib.pyplot as plt
        f = self.open_output('rowe134', wrapper=True)
        for i in 1,3,4:
            theta, xi, err = rowe_stats[i]
            plt.errorbar(theta, abs(xi), err, fmt='.', label=rf'$\rho_{i}$', capsize=3)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\xi_+(\theta)$")
        plt.legend()
        f.close()

        f = self.open_output('rowe25', wrapper=True)
        for i in 2,5:
            theta, xi, err = rowe_stats[i]
            plt.errorbar(theta, abs(xi), err, fmt='.', label=rf'$\rho_{i}$', capsize=3)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\xi_+(\theta)$")
        plt.legend()
        f.close()

    def save_stats(self, rowe_stats):
        f = self.open_output('rowe_stats')
        g = f.create_group("rowe_statistics")
        for i in 1,2,3,4,5:
            theta, xi, err = rowe_stats[i]
            h = g.create_group(f'rowe_{i}')
            h.create_dataset('theta', data=theta)
            h.create_dataset('xi_plus', data=xi)
            h.create_dataset('xi_err', data=err)
        f.close()




class TXBrighterFatterPlot(PipelineStage):
    name = 'TXBrighterFatterPlot'

    inputs =[('star_catalog', HDFFile)]

    outputs = [
        ('brighter_fatter', PNGFile),
        ('brighter_fatter_plot_data', HDFFile),
    ]

    config_options = {
        'band': 'r',
        'nbin': 20,
        'mmin': 18.5,
        'mmax': 23.5,
    }

    def run(self):
        import h5py
        import matplotlib
        matplotlib.use('agg')

        data = self.load_stars()
        results = self.compute_binned_stats(data)

        self.save_stats(results)
        self.save_plots(results)

    def load_stars(self):
        f = self.open_input('star_catalog')
        g = f['stars']

        band = self.config['band']
        data = {}
        data['mag'] = g[f'{band}_mag'][:]
        data['delta_e1'] = g['measured_e1'][:] - g['model_e1'][:]
        data['delta_e2'] = g['measured_e2'][:] - g['model_e2'][:]
        data['delta_T'] = g['measured_T'][:] - g['model_T'][:]

        return data

    def compute_binned_stats(self, data):
        mag = data['mag']
        mmin = self.config['mmin']
        mmax = self.config['mmax']
        nbin = self.config['nbin']
        edges = np.linspace(mmin, mmax, nbin+1)
        index = np.digitize(mag, edges)
        dT = np.zeros(nbin)
        errT = np.zeros(nbin)
        e1 = np.zeros(nbin)
        e2 = np.zeros(nbin)
        err1 = np.zeros(nbin)
        err2 = np.zeros(nbin)
        m = np.zeros(nbin)


        for i in range(nbin):
            w = np.where((index==i+1) & np.isfinite(data['delta_T']) & np.isfinite(data['delta_e1']) & np.isfinite(data['delta_e2']) )
            m[i] = mag[w].mean()
            dT_i = data['delta_T'][w]
            e1_i = data['delta_e1'][w]
            e2_i = data['delta_e2'][w]
            dT[i] = dT_i.mean()
            errT[i] = dT_i.std() / np.sqrt(dT_i.size)
            e1[i] = e1_i.mean()
            err1[i] = e1_i.std() / np.sqrt(e1_i.size)
            e2[i] = e2_i.mean()
            err2[i] = e2_i.std() / np.sqrt(e2_i.size)

        return [m, dT, errT, e1, err1, e2, err2]

    def save_plots(self, results):
        import matplotlib.pyplot as plt
        m, dT, errT, e1, err1, e2, err2 = results
        band = self.config['band']
        f = self.open_output('brighter_fatter', wrapper=True, figsize=(6,8))
        ax = plt.subplot(2,1,1)
        plt.errorbar(m, dT, errT, fmt='.')
        plt.ylabel(r"$T_\mathrm{PSF} - T_\mathrm{model}$ ($\mathrm{arcsec}^2$)")
        plt.ylim(-0.025, 0.1)
        plt.subplot(2,1,2, sharex=ax)
        plt.errorbar(m, e1, err1, label='$e_1$', fmt='.')
        plt.errorbar(m, e2, err2, label='$e_2$', fmt='.')
        plt.ylabel(r"$e_\mathrm{PSF} - e_\mathrm{model}$")
        plt.xlabel(f"{band}-band magnitude")
        plt.ylim(-0.02, 0.02)
        plt.legend()
        plt.tight_layout()
        f.close()

    def save_stats(self, results):
        (m, dT, errT, e1, err1, e2, err2) = results
        f = self.open_output('brighter_fatter_plot_data')
        g = f.create_group('brighter_fatter')
        g.attrs['band'] = self.config['band']
        g.create_dataset('mag', data=m)
        g.create_dataset('delta_T', data=dT)
        g.create_dataset('delta_T_error', data=errT)
        g.create_dataset('delta_e1', data=e1)
        g.create_dataset('delta_e1_error', data=err1)
        g.create_dataset('delta_e2', data=e2)
        g.create_dataset('delta_e2_error', data=err2)
        f.close()
