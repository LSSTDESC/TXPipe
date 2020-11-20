from .base_stage import PipelineStage
from .data_types import Directory, ShearCatalog, HDFFile, PNGFile, TomographyCatalog, RandomsCatalog, YamlFile
from parallel_statistics import ParallelHistogram, ParallelMeanVariance
import numpy as np
from .utils.calibration_tools import read_shear_catalog_type
from .utils.calibration_tools import apply_metacal_response, apply_lensfit_calibration
from .plotting import manual_step_histogram

STAR_PSF_USED = 0
STAR_PSF_RESERVED = 1
STAR_TYPES = [STAR_PSF_USED, STAR_PSF_RESERVED]
STAR_TYPE_NAMES = ['PSF-used', 'PSF-reserved']
STAR_COLORS = ['blue', 'orange', 'green']

class TXPSFDiagnostics(PipelineStage):
    """
    """
    name='TXPSFDiagnostics'

    inputs = [
        ('star_catalog', HDFFile)
    ]
    outputs = [
        ('e1_psf_residual_hist', PNGFile),
        ('e2_psf_residual_hist', PNGFile),
        ('T_psf_residual_hist', PNGFile),
        ('star_psf_stats', YamlFile),

    ]
    config = {}

    def run(self):
        # PSF tests
        import matplotlib
        matplotlib.use('agg')

        # Make plotters - in each case we supply the function to make
        # the thing we want to histogram, the name, label, and range
        plotters = [
            self.plot_histogram(
                lambda d: (d['measured_e1'] - d['model_e1']),
                'e1_psf_residual',
                '$(e_{1}-e_{1,psf})$',
                np.linspace(-0.1, 0.1, 51),
                ),

            self.plot_histogram(
                lambda d: (d['measured_e2'] - d['model_e2']),
                'e2_psf_residual',
                '$(e_{2}-e_{2,psf})$',
                np.linspace(-0.1, 0.1, 51),
                ),

            self.plot_histogram(
                lambda d: (d['measured_T'] - d['model_T']) / d['measured_T'],
                'T_psf_residual',
                '$(T-T{psf})/T{psf}$',
                np.linspace(-0.1, 0.1, 51),
                ),

        ]

        # Start off each of the plotters.  This will make them all run up to the
        # first yield statement, then pause and wait for the first chunk of data
        for plotter in plotters:
            print(plotter)
            plotter.send(None)

        # Create an iterator for reading through the input data.
        # This method automatically splits up data among the processes,
        # so the plotters should handle this.
        chunk_rows = 10000
        star_cols = ['measured_e1',
                     'model_e1',
                     'measured_e2',
                     'model_e2',
                     'measured_T',
                     'model_T',
                     'calib_psf_used',
                     'calib_psf_reserved',
                     ]
        iter_star = self.iterate_hdf('star_catalog','stars',star_cols,chunk_rows)

        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for (start, end, data), in zip(iter_star):
            print(f"Read data {start} - {end}")
            data['star_type'] = load_star_type(data)
            # This causes each data = yield statement in each plotter to
            # be given this data chunk as the variable data.
            #data.update(data2)
            #data.update(data3)
            for plotter in plotters:
                plotter.send(data)

        # Tell all the plotters to finish, collect together results from the different
        # processors, and make their final plots.  Plotters need to respond
        # to the None input and
        results = {}
        for plotter in plotters:
            try:
                plotter.send(None)
            except StopIteration as err:
                # This feels like some cruel abuse because I don't
                # really understand how co-routines should be used.
                results.update(err.value)

        # save all counts, means, and sigmas to yaml file
        with self.open_output("star_psf_stats", wrapper=True) as f:
            f.write(results)

    def plot_histogram(self, function, output_name, xlabel, edges):
        import matplotlib.pyplot as plt
        print(f"Plotting {output_name}")
        counters = {s:ParallelHistogram(edges) for s in STAR_TYPES}
        stats = ParallelMeanVariance(len(STAR_TYPES))

        while True:
            data = yield

            if data is None:
                break

            # Get the specific quantity to plot
            # from the full data set
            value = function(data)

            for s in STAR_TYPES:
                r = data['star_type'] == s
                # build up histogram
                counters[s].add_data(value[r])
                # Also record data for mean and variance calculation
                stats.add_data(s, value[r])

        counts = {}
        for s in STAR_TYPES:
            counts[s] = counters[s].collect(self.comm)

        fig = self.open_output(output_name + "_hist", wrapper=True, figsize=(6,15))
        for s in STAR_TYPES:
            plt.subplot(len(STAR_TYPES), 1, s+1)
            manual_step_histogram(edges,
                                  counts[s],
                                  color=STAR_COLORS[s],
                                  label=STAR_TYPE_NAMES[s]
            )
            plt.title(STAR_TYPE_NAMES[s])
            plt.xlabel(xlabel)
            plt.ylabel(r'$N_{stars}$')
        fig.close()

        n, mu, sigma2 = stats.collect(self.comm)
        results = {}
        for s in STAR_TYPES:
            name = STAR_TYPE_NAMES[s]
            results[f'{output_name}_{name}_n'] = int(n[s])
            results[f'{output_name}_{name}_mu'] =float(mu[s])
            results[f'{output_name}_{name}_std'] = float(sigma2[s]**0.5)

        return results


class TXRoweStatistics(PipelineStage):
    """
    People sometimes think that these statistics are called the Rho statistics,
    because we usually use that letter for them.  Not so.  They are named after
    the wonderfully incorrigible rogue Barney Rowe, now sadly lost to high finance,
    who presented the first two of them in MNRAS 404, 350 (2010).
    """

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
        'psf_size_units':'sigma',
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib
        matplotlib.use('agg')

        ra, dec, e_psf, de_psf, T_f, star_type = self.load_stars()

        rowe_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            rowe_stats[1, t] = self.compute_rowe(1, s, ra, dec, de_psf,       de_psf)
            rowe_stats[2, t] = self.compute_rowe(2, s, ra, dec, e_psf,        de_psf)
            rowe_stats[3, t] = self.compute_rowe(3, s, ra, dec, e_psf*T_f, e_psf*T_f)
            rowe_stats[4, t] = self.compute_rowe(4, s, ra, dec, de_psf,    e_psf*T_f)
            rowe_stats[5, t] = self.compute_rowe(5, s, ra, dec, e_psf,     e_psf*T_f)

        self.save_stats(rowe_stats)
        self.rowe_plots(rowe_stats)



    def load_stars(self):
        with self.open_input('star_catalog') as f:
            g = f['stars']
            ra = g['ra'][:]
            dec = g['dec'][:]
            e1 = g['measured_e1'][:]
            e2 = g['measured_e2'][:]
            de1 = e1 - g['model_e1'][:]
            de2 = e2 - g['model_e2'][:]
            if self.config['psf_size_units']=='T':
                T_frac = (g['measured_T'][:] - g['model_T'][:]) / g['measured_T'][:]
            elif self.config['psf_size_units']=='sigma':
                T_frac = (g['measured_T'][:]**2 - g['model_T'][:]**2) / g['measured_T'][:]**2

            e_psf = np.array((e1, e2))
            de_psf = np.array((de1, de2))

            star_type = load_star_type(g)

        return ra, dec, e_psf, de_psf, T_frac, star_type

    def compute_rowe(self, i, s, ra, dec, q1, q2):
        # select a subset of the stars
        ra = ra[s]
        dec = dec[s]
        q1 = q1[:, s]
        q2 = q2[:, s]
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
        import matplotlib.transforms as mtrans

        f = self.open_output('rowe134', wrapper=True, figsize=(10, 6*len(STAR_TYPES)))
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s+1)


            for j,i in enumerate([1,3,4]):
                theta, xi, err = rowe_stats[i, s]
                tr = mtrans.offset_copy(ax.transData, f.file, 0.05*(j-1), 0, units='inches')
                plt.errorbar(theta, abs(xi), err, fmt='.', label=rf'$\rho_{i}$', capsize=3, transform=tr)
            plt.bar(0.0,2e-05,width=5,align='edge',color='gray',alpha=0.2)
            plt.bar(5,1e-07,width=245,align='edge',color='gray',alpha=0.2)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$\xi_+(\theta)$")
            plt.legend()
            plt.title(STAR_TYPE_NAMES[s])
        f.close()

        f = self.open_output('rowe25', wrapper=True, figsize=(10, 6*len(STAR_TYPES)))
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s+1)
            for j,i in enumerate([2,5]):
                theta, xi, err = rowe_stats[i, s]
                tr = mtrans.offset_copy(ax.transData, f.file, 0.05*j-0.025, 0, units='inches')
                plt.errorbar(theta, abs(xi), err, fmt='.', label=rf'$\rho_{i}$', capsize=3, transform=tr)
                plt.title(STAR_TYPE_NAMES[s])
                plt.bar(0.0,2e-05,width=5,align='edge',color='gray',alpha=0.2)
                plt.bar(5,1e-07,width=245,align='edge',color='gray',alpha=0.2)
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
            for s in STAR_TYPES:
                theta, xi, err = rowe_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f'rowe_{i}_{name}')
                h.create_dataset('theta', data=theta)
                h.create_dataset('xi_plus', data=xi)
                h.create_dataset('xi_err', data=err)
        f.close()

class TXStarShearTests(PipelineStage):
    """
    Star cross galaxy and star autocorrelation.
    """

    name = 'TXGalaxyStarShear'

    inputs =[('shear_catalog', ShearCatalog),('star_catalog', HDFFile),
    ('shear_tomography_catalog', TomographyCatalog)]
    outputs = [
        ('star_shear_test', PNGFile),
        ('star_star_test', PNGFile),
        ('star_shear_stats', HDFFile),
    ]

    config_options = {
        'min_sep':0.5,
        'max_sep':250.0,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'psf_size_units':'sigma',
        'shear_catalog_type':'metacal',
        'flip_g2': False,
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib
        matplotlib.use('agg')

        ra, dec, e_psf, de_psf, star_type = self.load_stars()
        ra_gal, dec_gal, g1, g2, weight = self.load_galaxies()

        #only use reserved stars for this statistics
        galaxy_star_stats = {}
        star_star_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            galaxy_star_stats[1, t] = self.compute_galaxy_star(ra, dec, e_psf, s, ra_gal, dec_gal, g1, g2, weight)
            galaxy_star_stats[2, t] = self.compute_galaxy_star(ra, dec, de_psf, s, ra_gal, dec_gal, g1, g2, weight)
            star_star_stats[1, t] = self.compute_star_star(ra, dec, e_psf, s, ra_gal, dec_gal, e_psf, weight)
            star_star_stats[2, t] = self.compute_star_star(ra, dec, de_psf, s, ra_gal, dec_gal, de_psf, weight)

        self.save_stats(galaxy_star_stats,star_star_stats)
        self.galaxy_star_plots(galaxy_star_stats)
        self.star_star_plots(star_star_stats)



    def load_stars(self):
        with self.open_input('star_catalog') as f:
            g = f['stars']
            ra = g['ra'][:]
            dec = g['dec'][:]
            e1 = g['measured_e1'][:]
            e2 = g['measured_e2'][:]
            de1 = e1 - g['model_e1'][:]
            de2 = e2 - g['model_e2'][:]

            e_psf = np.array((e1, e2))
            de_psf = np.array((de1, de2))

            star_type = load_star_type(g)

        return ra, dec, e_psf, de_psf, star_type

    def load_galaxies(self):

        # Columns we need from the shear catalog
        # TODO: not sure of an application where we would want to use true shear but can be added
        read_shear_catalog_type(self)

        # load tomography data
        f = self.open_input('shear_tomography_catalog')
        source_bin = f['tomography/source_bin'][:]
        mask = (source_bin!=-1) # Only use the sources that pass the fiducial cuts
        if self.config['shear_catalog_type']=='metacal':
            R_total_2d = f['metacal_response/R_total_2d'][:]

        f = self.open_input('shear_catalog')
        g = f['shear']
        ra = g['ra'][:][mask]
        dec = g['dec'][:][mask]
        if self.config['shear_catalog_type']=='metacal':
            g1 = g['mcal_g1'][:][mask]
            g2 = g['mcal_g2'][:][mask]
            weight = g['weight'][:][mask]
        else:
            g1 = g['g1'][:][mask]
            g2 = g['g2'][:][mask]
            weight = g['weight'][:][mask]
            sigma_e = g['sigma_e'][:][mask]
            m = g['m'][:][mask]

        if self.config['flip_g2']:
            g2 *= -1

        if self.config['shear_catalog_type']=='metacal':
            # We use S=0 here because we have already included it in R_total
            g1, g2 = apply_metacal_response(R_total_2d, 0.0, g1, g2)

        elif self.config['shear_catalog_type']=='lensfit':
            g1, g2, weight, _ = apply_lensfit_calibration(g1 = g1,g2 = g2,weight = weight,sigma_e = sigma_e, m = m)
        else:
            print('Shear calibration type not recognized.')

        return ra, dec, g1, g2, weight

    def compute_galaxy_star(self, ra, dec, q, s, ra_gal, dec_gal, g1, g2, weight):
        # select the star type
        ra = ra[s]
        dec = dec[s]
        q = q[:, s] # PSF quantity, either ellipticity or residual
        n = len(ra)
        i = len(ra_gal)
        print(f"Computing galaxy-cross-star statistic from {n} stars and {i} galaxies")
        import treecorr
        corr = treecorr.GGCorrelation(self.config)
        cat1 = treecorr.Catalog(ra=ra, dec=dec, g1=q[0], g2=q[1], ra_units='deg', dec_units='deg')
        cat2 = treecorr.Catalog(ra=ra_gal, dec=dec_gal, g1=g2, g2=g2, ra_units='deg', dec_units='deg', w=weight)
        corr.process(cat1, cat2)
        return corr.meanr, corr.xip, corr.varxip**0.5

    def compute_star_star(self, ra, dec, q1, s, ra_gal, dec_gal, q2, weight):
        # select the star type
        ra = ra[s]
        dec = dec[s]
        q1 = q1[:, s]
        q2 = q2[:, s]
        n = len(ra)
        i = len(ra_gal)
        print(f"Computing galaxy-cross-star statistic from {n} stars and {i} galaxies")
        import treecorr
        corr = treecorr.GGCorrelation(self.config)
        cat1 = treecorr.Catalog(ra=ra, dec=dec, g1=q1[0], g2=q1[1], ra_units='deg', dec_units='deg')
        cat2 = treecorr.Catalog(ra=ra, dec=dec, g1=q2[0], g2=q2[1], ra_units='deg', dec_units='deg')
        corr.process(cat1, cat2)
        return corr.meanr, corr.xip, corr.varxip**0.5

    def galaxy_star_plots(self, galaxy_star_stats):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans

        f = self.open_output('star_shear_test', wrapper=True, figsize=(10, 6*len(STAR_TYPES)))
        TEST_TYPES = ['shear','residual']
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s+1)
            for j,i in enumerate([1,2]):
                theta, xi, err = galaxy_star_stats[i, s]
                tr = mtrans.offset_copy(ax.transData, f.file, 0.05*j-0.025, 0, units='inches')
                plt.errorbar(theta, abs(xi), err, fmt='.', label=f'galaxy cross star {TEST_TYPES[i-1]}', capsize=3, transform=tr)
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi_+(\theta)$")
                plt.legend()
        f.close()

    def star_star_plots(self, star_star_stats):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans

        f = self.open_output('star_star_test', wrapper=True, figsize=(10, 6*len(STAR_TYPES)))
        TEST_TYPES = ['shear','residual']
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s+1)
            for j,i in enumerate([1,2]):
                theta, xi, err = star_star_stats[i, s]
                tr = mtrans.offset_copy(ax.transData, f.file, 0.05*j-0.025, 0, units='inches')
                plt.errorbar(theta, abs(xi), err, fmt='.', label=f'star cross star {TEST_TYPES[i-1]}', capsize=3, transform=tr)
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi_+(\theta)$")
                plt.legend()
        f.close()

    def save_stats(self, galaxy_star_stats, star_star_stats):

        f = self.open_output('star_shear_stats')
        g = f.create_group("star_cross_galaxy")
        for i in 1,2:
            for s in STAR_TYPES:
                theta, xi, err = galaxy_star_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f'star_cross_galaxy_{i}_{name}')
                h.create_dataset('theta', data=theta)
                h.create_dataset('xi_plus', data=xi)
                h.create_dataset('xi_err', data=err)

        g = f.create_group("star_cross_star")
        for i in 1,2:
            for s in STAR_TYPES:
                theta, xi, err = star_star_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f'star_cross_star_{i}_{name}')
                h.create_dataset('theta', data=theta)
                h.create_dataset('xi_plus', data=xi)
                h.create_dataset('xi_err', data=err)
        f.close()




class TXStarDensityTests(PipelineStage):
    """
    Star cross galaxy and star autocorrelation density stats.
    """

    name = 'TXGalaxyStarDensity'

    inputs =[('shear_catalog', ShearCatalog),('star_catalog', HDFFile),
    ('shear_tomography_catalog', TomographyCatalog),
    ('random_cats', RandomsCatalog)]
    outputs = [
        ('star_density_test', PNGFile),
        ('star_density_stats', HDFFile),
    ]

    config_options = {
        'min_sep':0.5,
        'max_sep':250.0,
        'nbins':20,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'psf_size_units':'sigma',
        'flip_g2': False,
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib
        matplotlib.use('agg')

        ra, dec,  star_type = self.load_stars()
        ra_gal, dec_gal = self.load_galaxies()
        ra_random, dec_random = self.load_randoms()

        galaxy_star_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            galaxy_star_stats[1, t] = self.compute_galaxy_star(ra, dec, s, ra_gal, dec_gal, ra_random, dec_random)
            galaxy_star_stats[2, t] = self.compute_star_star(ra, dec, s, ra_gal, dec_gal, ra_random, dec_random)

        self.save_stats(galaxy_star_stats)
        self.galaxy_star_plots(galaxy_star_stats)



    def load_stars(self):
        with self.open_input('star_catalog') as f:
            g = f['stars']
            ra = g['ra'][:]
            dec = g['dec'][:]

            star_type = load_star_type(g)

        return ra, dec, star_type

    def load_randoms(self):

        with self.open_input('random_cats') as f:
            group = f['randoms']
            ra_random = group['ra'][:]
            dec_random = group['dec'][:]
        return ra_random, dec_random

    def load_galaxies(self):

        # Columns we need from the shear catalog
        # TODO: not sure of an application where we would want to use true shear but can be added
        read_shear_catalog_type(self)

        # load tomography data
        with self.open_input('shear_tomography_catalog') as f:
            source_bin = f['tomography/source_bin'][:]
            mask = (source_bin!=-1) # Only use the sources that pass the fiducial cuts

        with self.open_input('shear_catalog') as f:
            g = f['shear']
            ra = g['ra'][:][mask]
            dec = g['dec'][:][mask]

        return ra, dec

    def compute_galaxy_star(self, ra, dec, s, ra_gal, dec_gal, ra_random, dec_random):
        # select the star type
        ra = ra[s]
        dec = dec[s]
        n = len(ra)
        i = len(ra_gal)
        print(f"Computing galaxy-cross-star statistic from {n} stars and {i} galaxies")

        import treecorr

        rancat  = treecorr.Catalog(
                ra=ra_random, dec=dec_random,
                ra_units='degree', dec_units='degree')

        cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')
        cat2 = treecorr.Catalog(ra=ra_gal, dec=dec_gal, ra_units='deg', dec_units='deg')

        nn = treecorr.NNCorrelation(self.config)
        rn = treecorr.NNCorrelation(self.config)
        nr = treecorr.NNCorrelation(self.config)
        rr = treecorr.NNCorrelation(self.config)

        nn.process(cat1,    cat2)
        nr.process(cat1,    rancat)
        rn.process(rancat, cat2)
        rr.process(rancat, rancat)

        nn.calculateXi(rr, dr=nr, rd=rn)
        return nn.meanr, nn.xi, nn.varxi**0.5

    def compute_star_star(self, ra, dec, s, ra_gal, dec_gal, ra_random, dec_random):
        # select the star type
        ra = ra[s]
        dec = dec[s]
        n = len(ra)
        i = len(ra_gal)
        print(f"Computing galaxy-cross-star statistic from {n} stars and {i} galaxies")

        import treecorr

        rancat  = treecorr.Catalog(
                ra=ra_random, dec=dec_random,
                ra_units='degree', dec_units='degree')

        cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')
        cat2 = treecorr.Catalog(ra=ra_gal, dec=dec_gal, ra_units='deg', dec_units='deg')

        nn = treecorr.NNCorrelation(self.config)
        rn = treecorr.NNCorrelation(self.config)
        nr = treecorr.NNCorrelation(self.config)
        rr = treecorr.NNCorrelation(self.config)

        nn.process(cat1,    cat1)
        nr.process(cat1,    rancat)
        rn.process(rancat, cat1)
        rr.process(rancat, rancat)

        nn.calculateXi(rr, dr=nr, rd=rn)
        return nn.meanr, nn.xi, nn.varxi**0.5

    def galaxy_star_plots(self, galaxy_star_stats):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans

        f = self.open_output('star_density_test', wrapper=True, figsize=(10, 6*len(STAR_TYPES)))
        TEST_TYPES = ['star cross galaxy','star cross star']
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s+1)
            for j,i in enumerate([1,2]):
                theta, xi, err = galaxy_star_stats[i, s]
                tr = mtrans.offset_copy(ax.transData, f.file, 0.05*j-0.025, 0, units='inches')
                plt.errorbar(theta, abs(xi), err, fmt='.', label=f'{TEST_TYPES[i-1]} density stats', capsize=3, transform=tr)
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi_+(\theta)$")
                plt.legend()
        f.close()

    def save_stats(self, galaxy_star_stats):

        f = self.open_output('star_density_stats')
        g = f.create_group("star_density")
        for i in 1,2:
            for s in STAR_TYPES:
                theta, xi, err = galaxy_star_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f'star_density_{i}_{name}')
                h.create_dataset('theta', data=theta)
                h.create_dataset('xi_plus', data=xi)
                h.create_dataset('xi_err', data=err)
        f.close()

class TXBrighterFatterPlot(PipelineStage):
    name = 'TXBrighterFatterPlot'

    inputs =[('star_catalog', HDFFile)]

    outputs = [
        ('brighter_fatter_plot', PNGFile),
        ('brighter_fatter_data', HDFFile),
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
        results = {}
        for s in STAR_TYPES:
            w = data['star_type'] == s
            data_s = {k: v[w] for k, v in data.items()}
            results[s] = self.compute_binned_stats(data_s)

        self.save_stats(results)
        self.save_plots(results)

    def load_stars(self):
        with self.open_input('star_catalog') as f:
            g = f['stars']

            band = self.config['band']
            data = {}
            data['mag'] = g[f'mag_{band}'][:]
            data['delta_e1'] = g['measured_e1'][:] - g['model_e1'][:]
            data['delta_e2'] = g['measured_e2'][:] - g['model_e2'][:]
            data['delta_T'] = g['measured_T'][:] - g['model_T'][:]
            data['star_type'] = load_star_type(g)

        return data

    def compute_binned_stats(self, data):
        # Which band this corresponds to depends on the
        # configuration option chosen
        mag = data['mag']
        mmin = self.config['mmin']
        mmax = self.config['mmax']
        nbin = self.config['nbin']
        # bin edges in magnitude
        edges = np.linspace(mmin, mmax, nbin+1)
        index = np.digitize(mag, edges)

        # Space for all the output values to go in
        dT = np.zeros(nbin)
        errT = np.zeros(nbin)
        e1 = np.zeros(nbin)
        e2 = np.zeros(nbin)
        err1 = np.zeros(nbin)
        err2 = np.zeros(nbin)
        m = np.zeros(nbin)


        for i in range(nbin):
            # Select only objects where everything is finite, as well
            # as only thing in this tomographic bin
            w = np.where((index==i+1) & np.isfinite(data['delta_T']) & np.isfinite(data['delta_e1']) & np.isfinite(data['delta_e2']) )
            # x-value = mean mag
            m[i] = mag[w].mean()
            # y values
            dT_i = data['delta_T'][w]
            e1_i = data['delta_e1'][w]
            e2_i = data['delta_e2'][w]
            # Mean and error on mean of each of these quantities
            dT[i] = dT_i.mean()
            errT[i] = dT_i.std() / np.sqrt(dT_i.size)
            e1[i] = e1_i.mean()
            err1[i] = e1_i.std() / np.sqrt(e1_i.size)
            e2[i] = e2_i.mean()
            err2[i] = e2_i.std() / np.sqrt(e2_i.size)

        return [m, dT, errT, e1, err1, e2, err2]

    def save_plots(self, results):
        import matplotlib.pyplot as plt
        band = self.config['band']
        n = len(results)
        width = n * 6
        f = self.open_output('brighter_fatter_plot', wrapper=True, figsize=(width,8))
        for s, res in results.items():
            m, dT, errT, e1, err1, e2, err2 = res

            # Top plot - classic BF size plot, the size residual as a function of
            # magnitude
            ax = plt.subplot(2,n,2*s+1)
            plt.title(STAR_TYPE_NAMES[s])
            plt.errorbar(m, dT, errT, fmt='.')
            plt.xlabel(f"{band}-band magnitude")
            plt.ylabel(r"$T_\mathrm{PSF} - T_\mathrm{model}$ ($\mathrm{arcsec}^2$)")
            plt.ylim(-0.025, 0.1)
            # Lower plot - the e1 and e2 residuals as a function of mag
            plt.subplot(2,n,2*s+2, sharex=ax)
            plt.title(STAR_TYPE_NAMES[s])
            plt.errorbar(m, e1, err1, label='$e_1$', fmt='.')
            plt.errorbar(m, e2, err2, label='$e_2$', fmt='.')
            plt.ylabel(r"$e_\mathrm{PSF} - e_\mathrm{model}$")
            plt.xlabel(f"{band}-band magnitude")
            # May need to adjust this range
            plt.ylim(-0.02, 0.02)
        plt.legend()
        plt.tight_layout()
        f.close()

    def save_stats(self, results):
        # Save all the stats in results for later plotting
        # Save to standard HDF5 format.
        f = self.open_output('brighter_fatter_data')
        g1 = f.create_group('brighter_fatter')
        g1.attrs['band'] = self.config['band']
        for s, res in results.items():
            (m, dT, errT, e1, err1, e2, err2) = res
            g = g1.create_group(STAR_TYPE_NAMES[s])
            g.create_dataset('mag', data=m)
            g.create_dataset('delta_T', data=dT)
            g.create_dataset('delta_T_error', data=errT)
            g.create_dataset('delta_e1', data=e1)
            g.create_dataset('delta_e1_error', data=err1)
            g.create_dataset('delta_e2', data=e2)
            g.create_dataset('delta_e2_error', data=err2)
        f.close()



def load_star_type(data):
    used = data['calib_psf_used'][:]
    reserved = data['calib_psf_reserved'][:]

    star_type = np.zeros(used.size, dtype=int)
    star_type[used] = STAR_PSF_USED
    star_type[reserved] = STAR_PSF_RESERVED

    return star_type
