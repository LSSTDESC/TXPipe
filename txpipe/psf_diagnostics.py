from .base_stage import PipelineStage
from .data_types import (
    Directory,
    ShearCatalog,
    HDFFile,
    PNGFile,
    TomographyCatalog,
    RandomsCatalog,
    YamlFile,
)
from parallel_statistics import ParallelHistogram, ParallelMeanVariance
import numpy as np
from .utils.calibration_tools import read_shear_catalog_type
from .utils.calibration_tools import apply_metacal_response, apply_lensfit_calibration
from .plotting import manual_step_histogram

STAR_PSF_USED = 0
STAR_PSF_RESERVED = 1
STAR_TYPES = [STAR_PSF_USED, STAR_PSF_RESERVED]
STAR_TYPE_NAMES = ["PSF-used", "PSF-reserved"]
STAR_COLORS = ["blue", "orange", "green"]


class TXPSFDiagnostics(PipelineStage):
    """
    Make histograms of PSF values

    This makes histograms of e1 and e2 residuals, and of the fractional
    PSF size excess.
    """

    name = "TXPSFDiagnostics"

    inputs = [("star_catalog", HDFFile)]
    outputs = [
        ("e1_psf_residual_hist", PNGFile),
        ("e2_psf_residual_hist", PNGFile),
        ("T_frac_psf_residual_hist", PNGFile),
        ("star_psf_stats", YamlFile),
    ]
    config_options = {}

    def run(self):
        # PSF tests
        import matplotlib

        matplotlib.use("agg")

        # Make plotters - in each case we supply the function to make
        # the thing we want to histogram, the name, label, and range
        plotters = [
            self.plot_histogram(
                lambda d: (d["measured_e1"] - d["model_e1"]),
                "e1_psf_residual",
                "$e_{1}-e_{1,psf}$",
                np.linspace(-0.1, 0.1, 51),
            ),
            self.plot_histogram(
                lambda d: (d["measured_e2"] - d["model_e2"]),
                "e2_psf_residual",
                "$e_{2}-e_{2,psf}$",
                np.linspace(-0.1, 0.1, 51),
            ),
            self.plot_histogram(
                lambda d: (d["measured_T"] - d["model_T"]) / d["measured_T"],
                "T_frac_psf_residual",
                "$(T-T{psf})/T{psf}$",
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
        star_cols = [
            "measured_e1",
            "model_e1",
            "measured_e2",
            "model_e2",
            "measured_T",
            "model_T",
            "calib_psf_used",
            "calib_psf_reserved",
        ]
        iter_star = self.iterate_hdf("star_catalog", "stars", star_cols, chunk_rows)

        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for ((start, end, data),) in zip(iter_star):
            print(f"Read data {start} - {end}")
            data["star_type"] = load_star_type(data)
            # This causes each data = yield statement in each plotter to
            # be given this data chunk as the variable data.
            # data.update(data2)
            # data.update(data3)
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
        counters = {s: ParallelHistogram(edges) for s in STAR_TYPES}
        stats = ParallelMeanVariance(len(STAR_TYPES))

        while True:
            data = yield

            if data is None:
                break

            # Get the specific quantity to plot
            # from the full data set
            value = function(data)

            for s in STAR_TYPES:
                r = data["star_type"] == s
                # build up histogram
                d = value[r]
                counters[s].add_data(d)
                # Also record data for mean and variance calculation
                # There are some NaNs in here which are presumably
                # not propagated to PSF estimation anyway
                stats.add_data(s, d[np.isfinite(d)])

        counts = {}
        for s in STAR_TYPES:
            counts[s] = counters[s].collect(self.comm)

        fig = self.open_output(output_name + "_hist", wrapper=True, figsize=(6, 15))
        for s in STAR_TYPES:
            plt.subplot(len(STAR_TYPES), 1, s + 1)
            manual_step_histogram(
                edges, counts[s], color=STAR_COLORS[s], label=STAR_TYPE_NAMES[s]
            )
            plt.title(STAR_TYPE_NAMES[s])
            plt.xlabel(xlabel)
            plt.ylabel(r"$N_{stars}$")
        fig.close()

        n, mu, sigma2 = stats.collect(self.comm)
        results = {}
        for s in STAR_TYPES:
            name = STAR_TYPE_NAMES[s]
            results[f"{output_name}_{name}_n"] = int(n[s])
            results[f"{output_name}_{name}_mu"] = float(mu[s])
            results[f"{output_name}_{name}_std"] = float(sigma2[s] ** 0.5)

        return results


class TXTauStatistics(PipelineStage):
    """
    Compute and plot PSF Tau statistics where the definition of Tau stats are eq.20-22
    of Gatti et al 2023.
    """
    name     = "TXTauStatistics"
    parallel = False
    inputs   = [
                ("star_catalog", HDFFile),
                ("rowe"        , HDFFile),
               ]

    outputs  = [
                ("tau25"    , PNGFile),
                ("tau0"     , PNGFile),
                ("tau_stats", HDFFile),
               ]


    config_options = {
                       "min_sep"  : 0.5,
                       "max_sep"  : 250.0,
                       "nbins"    : 20,
                       "bin_slop" : 0.01,
                       "sep_units": "arcmin",
                       "psf_size_units": "sigma",
                     }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")

        ra, dec, e_psf, e_mod, de_psf, T_f, star_type = self.load_stars()
        
        rho0,rho1,rho2,rho3,rho4,rho5 = load_rowestats()

        tau_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            # Joint tau 0-2-5 data vector and cov
            # We need the off-diagonal 0-2-5 blocks
            tau,cov = self.compute_alltau(s, ra, dec, e_psf, e_mod, de_psf, T_f)

        # Run simple mcmc to find best-fit values for alpha,beta,eta
        a_best, a_best_err, b_best, b_best_err, n_best, n_best_err = sample(tau,cov,rho0,rho1,rho2,rho3,rho4,rho5)

        tau_stats[0, t] = alpha*rho0 + beta*rho2 + eta*rho5
        tau_stats[2, t] = alpha*rho2 + beta*rho1 + eta*rho4
        tau_stats[5, t] = alpha*rho5 + beta*rho4 + eta*rho3


    def compute_alltau(self, gra, gdec, g, s, sra, sdec, e_psf, e_mod, de_psf, T_f):
        '''
        Compute tau0, tau2, tau5.
        All three needs to be computed at once due to covariance.

        gra    : RA of galaxies
        gdec   : DEC of galaxies
        g      : shear for observed galaxies np.array((e1, e2))

        s      : indices of stars to use in calculation
        sra    : RA of stars
        sdec   : DEC of stars
        
        e_psf  : measured ellipticities of PSF from stars 
        e_mod  : model ellipticities of PSF
        de_psf : e_psf-e_mod
        T_f    : (T_meas - T_model)/T_meas
        
        '''
        #e_psf = np.array((e1psf, e2psf))
        #e_mod = np.array((e1mod,e2mod))
        #de_psf = np.array((de1, de2))
        
        p = e_mod
        w = de_psf
        p = e_psf * T_f
        
        import treecorr

        sra,sdec = sra[s],sdec[s] # Get ra/dec for  specific stars
                
        print(f"Computing Tau statistic rho_{i} from {n} objects")
        
        catg = treecorr.Catalog(ra=gra, dec=gdec, g1=g[0], g2=g[1], ra_units="deg", dec_units="deg") # galaxy shear
        catp = treecorr.Catalog(ra=sra, dec=sdec, g1=p[0], g2=p[1], ra_units="deg", dec_units="deg") # e_model
        catq = treecorr.Catalog(ra=sra, dec=sdec, g1=q[0], g2=q[1], ra_units="deg", dec_units="deg") # (e_* - e_model)
        catw = treecorr.Catalog(ra=sra, dec=sdec, g1=w[0], g2=w[1], ra_units="deg", dec_units="deg") # (e_*(T_* - T_model)/T_* )
            
        corr0 = treecorr.GGCorrelation(self.config)
        corr0.process(catg, catp)
        corr2 = treecorr.GGCorrelation(self.config)
        corr2.process(catg, catq)
        corr5 = treecorr.GGCorrelation(self.config)
        corr5.process(catg, catw)
        
        # Estimate covariance using jackknice
        cov = treecorr.estimate_multi_cov([corr0,corr2,corr5], 'jackknife')

        return corr.meanr, corr.xip, cov
    




    def sample(self,tau,cov,rho0,rho1,rho2,rho3,rho4,rho5, nwalkers=10, ndim=3):
        '''
        Run a simple mcmc chain to detemine the best-fit values for  
        '''
        import emcee

        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb, args=(x, y, yerr))
        sampler.run_mcmc(pos, 5000, progress=True);

        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

        ret = {}
        var = ['alpha','beta','eta']

        for i,v in enumerate(var):
           mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
           q    = np.diff(mcmc)
           ret[v] = {'mean': mcmc[1],'lerr': q[0], 'rerr': q[1]}

        return ret

    

    def logPrior(theta):
        ''''''
        m, b, log_f = theta
        if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
            return 0.0
        return -np.inf


    def logLike(self, X, alpha, beta, eta, cov):
        # Create combined template
        t_0    = alpha*rho0 + beta*rho2 + eta*rho5
        t_2    = alpha*rho2 + beta*rho1 + eta*rho4
        t_5    = alpha*rho5 + beta*rho4 + eta*rho3
        t_all  = np.concatenate([t_0,t_2,t_5])
 
        #Compute chi2
        chi2_0 = np.dot((X-t_all),np.dot(X-t_all,cov))

        return -0.5*chi2_0

    def logProb(theta, x, y, yerr):
        lp = logPrior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + loglike(theta, x, y, yerr)



class TXRoweStatistics(PipelineStage):
    """
    Compute and plot PSF Rowe statistics

    People sometimes think that these statistics are called the Rho statistics,
    because we usually use that letter for them.  Not so.  They are named after
    the wonderfully incorrigible rogue Barney Rowe, now sadly lost to high finance,
    who presented the first two of them in MNRAS 404, 350 (2010).
    """

    name = "TXRoweStatistics"
    parallel = False
    inputs = [("star_catalog", HDFFile)]
    outputs = [
        ("rowe134", PNGFile),
        ("rowe25", PNGFile),
        ("rowe0", PNGFile),
        ("rowe_stats", HDFFile),
    ]

    config_options = {
        "min_sep": 0.5,
        "max_sep": 250.0,
        "nbins": 20,
        "bin_slop": 0.01,
        "sep_units": "arcmin",
        "psf_size_units": "sigma",
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")

        ra, dec, e_psf, e_mod, de_psf, T_f, star_type = self.load_stars()

        rowe_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            rowe_stats[0, t] = self.compute_rowe(0, s, ra, dec, e_mod, e_mod)
            rowe_stats[1, t] = self.compute_rowe(1, s, ra, dec, de_psf, de_psf)
            rowe_stats[2, t] = self.compute_rowe(2, s, ra, dec, de_psf, e_mod)
            rowe_stats[3, t] = self.compute_rowe(3, s, ra, dec, e_psf * T_f, e_psf * T_f)
            rowe_stats[4, t] = self.compute_rowe(4, s, ra, dec, de_psf, e_psf * T_f)
            rowe_stats[5, t] = self.compute_rowe(5, s, ra, dec, e_mod, e_psf * T_f)

        self.save_stats(rowe_stats)
        self.rowe_plots(rowe_stats)

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g = f["stars"]
            ra = g["ra"][:]
            dec = g["dec"][:]
            e1psf = g["measured_e1"][:]
            e2psf = g["measured_e2"][:]
            e1mod = g["model_e1"][:]
            e2mod = g["model_e2"][:]
            de1 = e1psf - e1mod
            de2 = e2psf - e2mod
            if self.config["psf_size_units"] == "T":
                T_frac = (g["measured_T"][:] - g["model_T"][:]) / g["measured_T"][:]
            elif self.config["psf_size_units"] == "sigma":
                T_frac = (g["measured_T"][:] ** 2 - g["model_T"][:] ** 2) / g[
                    "measured_T"
                ][:] ** 2

            e_psf = np.array((e1psf, e2psf))
            e_mod = np.array((e1mod,e2mod))
            de_psf = np.array((de1, de2))

            star_type = load_star_type(g)

        return ra, dec, e_psf, e_mod, de_psf, T_frac, star_type

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
        cat1 = treecorr.Catalog(
            ra=ra, dec=dec, g1=q1[0], g2=q1[1], ra_units="deg", dec_units="deg"
        )
        cat2 = treecorr.Catalog(
            ra=ra, dec=dec, g1=q2[0], g2=q2[1], ra_units="deg", dec_units="deg"
        )
        corr.process(cat1, cat2)
        return corr.meanr, corr.xip, corr.varxip**0.5

    def rowe_plots(self, rowe_stats):
        # First plot - stats 1,3,4
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans
        
        f = self.open_output("rowe0",wrapper=True,figsize=(10,6*len(STAR_TYPES)))
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            
            for j,i in enumerate([0]):
                theta,xi,err = rowe_stats[i,s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * (j - 1), 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=rf"$\rho_{i}$",
                    capsize=3,
                    transform=tr,
                )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$\xi_+(\theta)$")
            plt.legend()
            plt.title(STAR_TYPE_NAMES[s])
        f.close()

        f = self.open_output("rowe134", wrapper=True, figsize=(10, 6 * len(STAR_TYPES)))
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)

            for j, i in enumerate([1, 3, 4]):
                theta, xi, err = rowe_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * (j - 1), 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=rf"$\rho_{i}$",
                    capsize=3,
                    transform=tr,
                )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$\xi_+(\theta)$")
            plt.legend()
            plt.title(STAR_TYPE_NAMES[s])
        f.close()

        f = self.open_output("rowe25", wrapper=True, figsize=(10, 6 * len(STAR_TYPES)))
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            for j, i in enumerate([2, 5]): 
                theta, xi, err = rowe_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * j - 0.025, 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=rf"$\rho_{i}$",
                    capsize=3,
                    transform=tr,
                )
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi_+(\theta)$")
                plt.legend()
        f.close()

    def save_stats(self, rowe_stats):
        f = self.open_output("rowe_stats")
        g = f.create_group("rowe_statistics")
        for i in 0, 1, 2, 3, 4, 5:
            for s in STAR_TYPES:
                theta, xi, err = rowe_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"rowe_{i}_{name}")
                h.create_dataset("theta", data=theta)
                h.create_dataset("xi_plus", data=xi)
                h.create_dataset("xi_err", data=err)
        f.close()


class TXGalaxyStarShear(PipelineStage):
    """
    Compute and plot star x galaxy and star x star correlations.

    These are shape correlations; they differ from the Rowe stats slightly
    because they measure star values not interpolated PSF values.
    """

    name = "TXGalaxyStarShear"
    parallel = False

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("star_catalog", HDFFile),
        ("shear_tomography_catalog", TomographyCatalog),
    ]
    outputs = [
        ("star_shear_test", PNGFile),
        ("star_star_test", PNGFile),
        ("star_shear_stats", HDFFile),
    ]

    config_options = {
        "min_sep": 0.5,
        "max_sep": 250.0,
        "nbins": 20,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "psf_size_units": "sigma",
        "shear_catalog_type": "metacal",
        "flip_g2": False,
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")

        ra, dec, e_psf, de_psf, star_type = self.load_stars()
        ra_gal, dec_gal, g1, g2, weight = self.load_galaxies()

        # only use reserved stars for this statistics
        galaxy_star_stats = {}
        star_star_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            galaxy_star_stats[1, t] = self.compute_galaxy_star(
                ra, dec, e_psf, s, ra_gal, dec_gal, g1, g2, weight
            )
            galaxy_star_stats[2, t] = self.compute_galaxy_star(
                ra, dec, de_psf, s, ra_gal, dec_gal, g1, g2, weight
            )
            star_star_stats[1, t] = self.compute_star_star(
                ra, dec, e_psf, s, ra_gal, dec_gal, e_psf, weight
            )
            star_star_stats[2, t] = self.compute_star_star(
                ra, dec, de_psf, s, ra_gal, dec_gal, de_psf, weight
            )

        self.save_stats(galaxy_star_stats, star_star_stats)
        self.galaxy_star_plots(galaxy_star_stats)
        self.star_star_plots(star_star_stats)

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g = f["stars"]
            ra = g["ra"][:]
            dec = g["dec"][:]
            e1 = g["measured_e1"][:]
            e2 = g["measured_e2"][:]
            de1 = e1 - g["model_e1"][:]
            de2 = e2 - g["model_e2"][:]

            e_psf = np.array((e1, e2))
            de_psf = np.array((de1, de2))

            star_type = load_star_type(g)

        return ra, dec, e_psf, de_psf, star_type

    def load_galaxies(self):

        # Columns we need from the shear catalog
        # TODO: not sure of an application where we would want to use true shear but can be added
        cat_type = read_shear_catalog_type(self)

        # load tomography data
        with self.open_input("shear_tomography_catalog") as f:
            source_bin = f["tomography/source_bin"][:]
            mask = source_bin != -1  # Only use the sources that pass the fiducial cuts
            if cat_type == "metacal":
                R_total_2d = f["response/R_total_2d"][:]
            elif cat_type == "metadetect":
                R_total_2d = f["response/R_2d"][:]

        with self.open_input("shear_catalog") as f:
            g = f["shear"]

            # Get the base catalog for metadetect
            if cat_type == "metadetect":
                g = g["00"]

            ra = g["ra"][:][mask]
            dec = g["dec"][:][mask]

            if cat_type == "metacal":
                g1 = g["mcal_g1"][:][mask]
                g2 = g["mcal_g2"][:][mask]
                weight = g["weight"][:][mask]

            elif cat_type == "metadetect":
                g1 = g["g1"][:][mask]
                g2 = g["g2"][:][mask]
                weight = g["weight"][:][mask]

            else:
                g1 = g["g1"][:][mask]
                g2 = g["g2"][:][mask]
                weight = g["weight"][:][mask]
                sigma_e = g["sigma_e"][:][mask]
                m = g["m"][:][mask]

        if self.config["flip_g2"]:
            g2 *= -1

        if cat_type == "metacal" or cat_type == "metadetect":
            # We use S=0 here because we have already included it in R_total
            g1, g2 = apply_metacal_response(R_total_2d, 0.0, g1, g2)

        elif cat_type == "lensfit":
            g1, g2, weight, _ = apply_lensfit_calibration(
                g1=g1, g2=g2, weight=weight, sigma_e=sigma_e, m=m
            )
        else:
            print("Shear calibration type not recognized.")

        return ra, dec, g1, g2, weight

    def compute_galaxy_star(self, ra, dec, q, s, ra_gal, dec_gal, g1, g2, weight):
        # select the star type
        ra = ra[s]
        dec = dec[s]
        q = q[:, s]  # PSF quantity, either ellipticity or residual
        n = len(ra)
        i = len(ra_gal)
        print(f"Computing galaxy-cross-star statistic from {n} stars and {i} galaxies")
        import treecorr

        corr = treecorr.GGCorrelation(self.config)
        cat1 = treecorr.Catalog(
            ra=ra, dec=dec, g1=q[0], g2=q[1], ra_units="deg", dec_units="deg"
        )
        cat2 = treecorr.Catalog(
            ra=ra_gal,
            dec=dec_gal,
            g1=g2,
            g2=g2,
            ra_units="deg",
            dec_units="deg",
            w=weight,
        )
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
        cat1 = treecorr.Catalog(
            ra=ra, dec=dec, g1=q1[0], g2=q1[1], ra_units="deg", dec_units="deg"
        )
        cat2 = treecorr.Catalog(
            ra=ra, dec=dec, g1=q2[0], g2=q2[1], ra_units="deg", dec_units="deg"
        )
        corr.process(cat1, cat2)
        return corr.meanr, corr.xip, corr.varxip**0.5

    def galaxy_star_plots(self, galaxy_star_stats):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans

        f = self.open_output(
            "star_shear_test", wrapper=True, figsize=(10, 6 * len(STAR_TYPES))
        )
        TEST_TYPES = ["shear", "residual"]
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            for j, i in enumerate([1, 2]):
                theta, xi, err = galaxy_star_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * j - 0.025, 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=f"galaxy cross star {TEST_TYPES[i-1]}",
                    capsize=3,
                    transform=tr,
                )
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi_+(\theta)$")
                plt.legend()
        f.close()

    def star_star_plots(self, star_star_stats):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans

        f = self.open_output(
            "star_star_test", wrapper=True, figsize=(10, 6 * len(STAR_TYPES))
        )
        TEST_TYPES = ["shear", "residual"]
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            for j, i in enumerate([1, 2]):
                theta, xi, err = star_star_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * j - 0.025, 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=f"star cross star {TEST_TYPES[i-1]}",
                    capsize=3,
                    transform=tr,
                )
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi_+(\theta)$")
                plt.legend()
        f.close()

    def save_stats(self, galaxy_star_stats, star_star_stats):

        f = self.open_output("star_shear_stats")
        g = f.create_group("star_cross_galaxy")
        for i in 1, 2:
            for s in STAR_TYPES:
                theta, xi, err = galaxy_star_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"star_cross_galaxy_{i}_{name}")
                h.create_dataset("theta", data=theta)
                h.create_dataset("xi_plus", data=xi)
                h.create_dataset("xi_err", data=err)

        g = f.create_group("star_cross_star")
        for i in 1, 2:
            for s in STAR_TYPES:
                theta, xi, err = star_star_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"star_cross_star_{i}_{name}")
                h.create_dataset("theta", data=theta)
                h.create_dataset("xi_plus", data=xi)
                h.create_dataset("xi_err", data=err)
        f.close()


class TXGalaxyStarDensity(PipelineStage):
    """
    Compute and plot star x galaxy and star x star density correlations

    This version uses the source catalog, though a version with lens
    samples might also be useful.
    """

    name = "TXGalaxyStarDensity"
    parallel = False

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("star_catalog", HDFFile),
        ("shear_tomography_catalog", TomographyCatalog),
        ("random_cats", RandomsCatalog),
    ]
    outputs = [
        ("star_density_test", PNGFile),
        ("star_density_stats", HDFFile),
    ]

    config_options = {
        "min_sep": 0.5,
        "max_sep": 250.0,
        "nbins": 20,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "psf_size_units": "sigma",
        "flip_g2": False,
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")

        ra, dec, star_type = self.load_stars()
        ra_gal, dec_gal = self.load_galaxies()
        ra_random, dec_random = self.load_randoms()

        galaxy_star_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            galaxy_star_stats[1, t] = self.compute_galaxy_star(
                ra, dec, s, ra_gal, dec_gal, ra_random, dec_random
            )
            galaxy_star_stats[2, t] = self.compute_star_star(
                ra, dec, s, ra_gal, dec_gal, ra_random, dec_random
            )

        self.save_stats(galaxy_star_stats)
        self.galaxy_star_plots(galaxy_star_stats)

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g = f["stars"]
            ra = g["ra"][:]
            dec = g["dec"][:]

            star_type = load_star_type(g)

        return ra, dec, star_type

    def load_randoms(self):

        with self.open_input("random_cats") as f:
            group = f["randoms"]
            ra_random = group["ra"][:]
            dec_random = group["dec"][:]
        return ra_random, dec_random

    def load_galaxies(self):

        # Columns we need from the shear catalog
        # TODO: not sure of an application where we would want to use true shear but can be added
        read_shear_catalog_type(self)

        # load tomography data
        with self.open_input("shear_tomography_catalog") as f:
            source_bin = f["tomography/source_bin"][:]
            mask = source_bin != -1  # Only use the sources that pass the fiducial cuts

        with self.open_input("shear_catalog", wrapper=True) as f:
            g = f.file["shear"]
            if f.catalog_type == "metadetect":
                g = g["00"]
            ra = g["ra"][:][mask]
            dec = g["dec"][:][mask]

        return ra, dec

    def compute_galaxy_star(self, ra, dec, s, ra_gal, dec_gal, ra_random, dec_random):
        # select the star type
        ra = ra[s]
        dec = dec[s]
        n = len(ra)
        i = len(ra_gal)
        print(f"Computing galaxy-cross-star statistic from {n} stars and {i} galaxies")

        import treecorr

        rancat = treecorr.Catalog(
            ra=ra_random, dec=dec_random, ra_units="degree", dec_units="degree"
        )

        cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units="deg", dec_units="deg")
        cat2 = treecorr.Catalog(ra=ra_gal, dec=dec_gal, ra_units="deg", dec_units="deg")

        nn = treecorr.NNCorrelation(self.config)
        rn = treecorr.NNCorrelation(self.config)
        nr = treecorr.NNCorrelation(self.config)
        rr = treecorr.NNCorrelation(self.config)

        nn.process(cat1, cat2)
        nr.process(cat1, rancat)
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

        rancat = treecorr.Catalog(
            ra=ra_random, dec=dec_random, ra_units="degree", dec_units="degree"
        )

        cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units="deg", dec_units="deg")
        cat2 = treecorr.Catalog(ra=ra_gal, dec=dec_gal, ra_units="deg", dec_units="deg")

        nn = treecorr.NNCorrelation(self.config)
        rn = treecorr.NNCorrelation(self.config)
        nr = treecorr.NNCorrelation(self.config)
        rr = treecorr.NNCorrelation(self.config)

        nn.process(cat1, cat1)
        nr.process(cat1, rancat)
        rn.process(rancat, cat1)
        rr.process(rancat, rancat)

        nn.calculateXi(rr, dr=nr, rd=rn)
        return nn.meanr, nn.xi, nn.varxi**0.5

    def galaxy_star_plots(self, galaxy_star_stats):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans

        f = self.open_output(
            "star_density_test", wrapper=True, figsize=(10, 6 * len(STAR_TYPES))
        )
        TEST_TYPES = ["star cross galaxy", "star cross star"]
        for s in STAR_TYPES:
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            for j, i in enumerate([1, 2]):
                theta, xi, err = galaxy_star_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * j - 0.025, 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=f"{TEST_TYPES[i-1]} density stats",
                    capsize=3,
                    transform=tr,
                )
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi_+(\theta)$")
                plt.legend()
        f.close()

    def save_stats(self, galaxy_star_stats):

        f = self.open_output("star_density_stats")
        g = f.create_group("star_density")
        for i in 1, 2:
            for s in STAR_TYPES:
                theta, xi, err = galaxy_star_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"star_density_{i}_{name}")
                h.create_dataset("theta", data=theta)
                h.create_dataset("xi_plus", data=xi)
                h.create_dataset("xi_err", data=err)
        f.close()


class TXBrighterFatterPlot(PipelineStage):
    """
    Compute and plot a diagnostic of the brighter-fatter effect

    This plots the mean e1, e2, and size difference between that interpolated
    at star locations and the star values themselves, as a function of magnitude.
    """

    name = "TXBrighterFatterPlot"
    parallel = False

    inputs = [("star_catalog", HDFFile)]

    outputs = [
        ("brighter_fatter_plot", PNGFile),
        ("brighter_fatter_data", HDFFile),
    ]

    config_options = {
        "band": "r",
        "nbin": 20,
        "mmin": 18.5,
        "mmax": 23.5,
    }

    def run(self):
        import h5py
        import matplotlib

        matplotlib.use("agg")

        data = self.load_stars()
        results = {}
        for s in STAR_TYPES:
            w = data["star_type"] == s
            data_s = {k: v[w] for k, v in data.items()}
            results[s] = self.compute_binned_stats(data_s)

        self.save_stats(results)
        self.save_plots(results)

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g = f["stars"]

            band = self.config["band"]
            data = {}
            data["mag"] = g[f"mag_{band}"][:]
            data["delta_e1"] = g["measured_e1"][:] - g["model_e1"][:]
            data["delta_e2"] = g["measured_e2"][:] - g["model_e2"][:]
            data["delta_T"] = g["measured_T"][:] - g["model_T"][:]
            data["star_type"] = load_star_type(g)

        return data

    def compute_binned_stats(self, data):
        # Which band this corresponds to depends on the
        # configuration option chosen
        mag = data["mag"]
        mmin = self.config["mmin"]
        mmax = self.config["mmax"]
        nbin = self.config["nbin"]
        # bin edges in magnitude
        edges = np.linspace(mmin, mmax, nbin + 1)
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
            w = np.where(
                (index == i + 1)
                & np.isfinite(data["delta_T"])
                & np.isfinite(data["delta_e1"])
                & np.isfinite(data["delta_e2"])
            )
            # x-value = mean mag
            m[i] = mag[w].mean()
            # y values
            dT_i = data["delta_T"][w]
            e1_i = data["delta_e1"][w]
            e2_i = data["delta_e2"][w]
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

        band = self.config["band"]
        n = len(results)
        width = n * 6
        f = self.open_output("brighter_fatter_plot", wrapper=True, figsize=(width, 8))
        for s, res in results.items():
            m, dT, errT, e1, err1, e2, err2 = res

            # Top plot - classic BF size plot, the size residual as a function of
            # magnitude
            ax = plt.subplot(2, n, 2 * s + 1)
            plt.title(STAR_TYPE_NAMES[s])
            plt.errorbar(m, dT, errT, fmt=".")
            plt.xlabel(f"{band}-band magnitude")
            plt.ylabel(r"$T_\mathrm{PSF} - T_\mathrm{model}$ ($\mathrm{arcsec}^2$)")
            plt.ylim(-0.025, 0.1)
            # Lower plot - the e1 and e2 residuals as a function of mag
            plt.subplot(2, n, 2 * s + 2, sharex=ax)
            plt.title(STAR_TYPE_NAMES[s])
            plt.errorbar(m, e1, err1, label="$e_1$", fmt=".")
            plt.errorbar(m, e2, err2, label="$e_2$", fmt=".")
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
        f = self.open_output("brighter_fatter_data")
        g1 = f.create_group("brighter_fatter")
        g1.attrs["band"] = self.config["band"]
        for s, res in results.items():
            (m, dT, errT, e1, err1, e2, err2) = res
            g = g1.create_group(STAR_TYPE_NAMES[s])
            g.create_dataset("mag", data=m)
            g.create_dataset("delta_T", data=dT)
            g.create_dataset("delta_T_error", data=errT)
            g.create_dataset("delta_e1", data=e1)
            g.create_dataset("delta_e1_error", data=err1)
            g.create_dataset("delta_e2", data=e2)
            g.create_dataset("delta_e2_error", data=err2)
        f.close()


def load_star_type(data):
    used = data["calib_psf_used"][:].astype('int')
    reserved = data["calib_psf_reserved"][:].astype('int')

    star_type = np.zeros(used.size, dtype=int)
    star_type[used] = STAR_PSF_USED
    star_type[reserved] = STAR_PSF_RESERVED

    return star_type
