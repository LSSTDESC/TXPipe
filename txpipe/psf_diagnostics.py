from .base_stage import PipelineStage
from .data_types import (
    Directory,
    ShearCatalog,
    HDFFile,
    PNGFile,
    TomographyCatalog,
    RandomsCatalog,
    YamlFile,
    TextFile,
    SACCFile,

)
from parallel_statistics import ParallelHistogram, ParallelMeanVariance
import numpy as np
import sys
import os
from .utils.calibration_tools import read_shear_catalog_type
from .plotting import manual_step_histogram
from .utils.calibrators import Calibrator


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

class TXPSFMomentCorr(PipelineStage):
    """
    Compute PSF Moments
    """

    name     = "TXPSFMomentCorr"
    parallel = False
    inputs   = [("star_catalog", HDFFile)]
    outputs  = [
                ("moments_stats", HDFFile)
               ]

    config_options = {
                      "min_sep"  : 0.5,
                      "max_sep"  : 250.0,
                      "nbins"    : 20,
                      "bin_slop" : 0.01,
                      "sep_units": "arcmin",
                      "subtract_mean" : False
                     }

    def run(self):
        import treecorr
        import h5py
        import matplotlib
        self.config["num_threads"] = int(os.environ.get("OMP_NUM_THREADS", 1))

        matplotlib.use("agg")

        # Load the star catalog
        ra, dec, e_meas, e_mod, de, moment4_meas, moment4_mod, dmoment4, star_type = self.load_stars()
        
        moments_stats = {}
        for t in STAR_TYPES:
            s = np.where(star_type==t)[0]
            moments_stats[0, t] = self.compute_momentcorr(0, s, ra, dec, e_mod, e_mod)
            moments_stats[1, t] = self.compute_momentcorr(1, s, ra, dec, moment4_mod, e_mod)
            moments_stats[2, t] = self.compute_momentcorr(2, s, ra, dec, moment4_mod, moment4_mod)
            moments_stats[3, t] = self.compute_momentcorr(3, s, ra, dec, de, e_mod)
            moments_stats[4, t] = self.compute_momentcorr(4, s, ra, dec, de, moment4_mod)
            moments_stats[5, t] = self.compute_momentcorr(5, s, ra, dec, dmoment4, moment4_mod)
            moments_stats[6, t] = self.compute_momentcorr(6, s, ra, dec, e_mod, dmoment4)
            moments_stats[7, t] = self.compute_momentcorr(7, s, ra, dec, de, de)
            moments_stats[8, t] = self.compute_momentcorr(8, s, ra, dec, de, dmoment4)
            moments_stats[9, t] = self.compute_momentcorr(9, s, ra, dec, dmoment4, dmoment4)
            
        self.save_stats(moments_stats)

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g         = f["stars"]
            ra        = g["ra"][:]
            dec       = g["dec"][:]
            e1meas    = g["measured_e1"][:]
            e2meas    = g["measured_e2"][:]
            e1mod     = g["model_e1"][:]
            e2mod     = g["model_e2"][:]
            e1meas_moment4 = g["measured_moment4_e1"][:]
            e2meas_moment4 = g["measured_moment4_e2"][:]
            e1mod_moment4  = g["model_moment4_e1"][:]
            e2mod_moment4  = g["model_moment4_e2"][:]
            
            # Note: definition are flipped for this paper
            de1         = e1mod - e1meas
            de2         = e2mod - e2meas 
            de1_moment4 = e1mod_moment4 - e1meas_moment4 
            de2_moment4 = e2mod_moment4 - e2meas_moment4 

            if self.config['subtract_mean']:
                e_meas       = np.array((e1meas-np.mean(e1meas), e2meas-np.mean(e2meas)))
                e_mod        = np.array((e1mod-np.mean(e1mod)  , e2mod-np.mean(e2mod)))
                de           = np.array((de1-np.mean(de1)      , de2-np.mean(de2)))
                moment4_meas = np.array((e1meas_moment4-np.mean(e1meas_moment4), e2meas_moment4-np.mean(e2meas_moment4)))
                moment4_mod  = np.array((e1mod_moment4-np.mean(e1mod_moment4)  , e2mod_moment4-np.mean(e2mod_moment4)))
                dmoment4     = np.array((de1_moment4-np.mean(de1_moment4)      , de2_moment4-np.mean(de2_moment4)))
            else:
                e_meas       = np.array((e1meas, e2meas ))
                e_mod        = np.array((e1mod , e2mod  ))
                de           = np.array((de1   , de2    ))
                moment4_meas = np.array((e1meas_moment4, e2meas_moment4))
                moment4_mod  = np.array((e1mod_moment4 , e2mod_moment4))
                dmoment4     = np.array((de1_moment4   , de2_moment4))

            star_type = load_star_type(g)

        return ra, dec, e_meas, e_mod, de, moment4_meas, moment4_mod, dmoment4, star_type

    def compute_momentcorr(self, i, s, ra, dec, q1, q2):
        # select a subset of the stars
        ra  = ra[s]
        dec = dec[s]
        q1  = q1[:, s]
        q2  = q2[:, s]
        n   = len(ra)
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

    def moment_plots(self, rowe_stats):
        # First plot - stats 1,3,4
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans
        
        f = self.open_output("moments",wrapper=True,figsize=(10,6*len(STAR_TYPES)))

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
                    # transform=tr,
                )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$\xi_+(\theta)$")
            plt.legend()
            plt.title(STAR_TYPE_NAMES[s])
        f.close()


    def save_stats(self, moments_stats):
        f = self.open_output("moments_stats")
        g = f.create_group("moment_statistics")
        for i in range(0,10):
            for s in STAR_TYPES:
                theta, xi, err = moments_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"moment4_{i}_{name}")
                h.create_dataset("theta"  , data=theta)
                h.create_dataset("xi_plus", data=xi)
                h.create_dataset("xi_err" , data=err)
        f.close()


class TXTauStatistics(PipelineStage):
    """
    Compute and plot PSF Tau statistics where the definition of Tau stats are eq. 20-22
    of Gatti et al 2023.
    """
    name     = "TXTauStatistics"
    parallel = False
    inputs   = [("binned_shear_catalog"    , ShearCatalog),
                ("star_catalog"            , HDFFile),
                ("rowe_stats"              , HDFFile),
               ]

    outputs  = [
                ("tau0"    , PNGFile), 
                ("tau2"    , PNGFile),
                ("tau5"    , PNGFile),
                ("tau_stats", HDFFile),
               ]


    config_options = {
                       "min_sep"       : 0.5,
                       "max_sep"       : 250.0,
                       "nbins"         : 20,
                       "bin_slop"      : 0.01,
                       "sep_units"     : "arcmin",
                       "npatch"        : 150,
                       "psf_size_units": "sigma",
                       "subtract_mean" : False,
                       "dec_cut"       : True,           # affects KiDS-1000 only
                       "star_type"     : 'PSF-reserved',
                       "cov_method"    : 'bootstrap',
                       "flip_g2"       : False,
                       "tomographic"   : True,
                     }

    def run(self):
        import treecorr
        import h5py
        import matplotlib
        import emcee

        matplotlib.use("agg")
        
        tau_stats  = {}
        p_bestfits = {}

        # Load star properties
        ra, dec, e_psf, e_mod, de_psf, T_f, star_type = self.load_stars()

        for s in STAR_TYPES:
            
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue

            # Load precomputed Rowe stats 
            rowe_stats = self.load_rowe(s)

            # set ranges for mcmc 
            ranges = {}
            ranges['alpha'] = [-1000.00, 1000.00]
            ranges['beta']  = [-1000.00, 1000.00]
            ranges['eta']   = [-1000.00, 1000.00]
            
            tau_stats[s]  = {}
            p_bestfits[s] = {}
            # Load galaxies
            if self.config['tomographic']:
                with self.open_input("binned_shear_catalog") as f:
                     nzbin = list(range(f["shear"].attrs["nbin_source"])) + ['all']
            else:
                nzbin = ['all']
            print(nzbin)
            for n in nzbin:
                gal_ra, gal_dec, gal_g1, gal_g2, gal_weight = self.load_galaxies(n)
                gal_g = np.array((gal_g1, gal_g2))
                print(f"computing tau stats for bin {n}")
                # Joint tau 0-2-5 data vector and cov
                # tau_stats contains [ang, tau0, tau2, tau5, cov]
                tau_stats[s][f'bin_{n}'] = self.compute_all_tau(gal_ra, gal_dec, gal_g, gal_weight, 
                                                    s, ra, dec, e_psf, e_mod, de_psf, T_f, star_type)

                # Run simple mcmc to find best-fit values for alpha,beta,eta
                p_bestfits[s][f'bin_{n}'] = self.sample(tau_stats[s][f'bin_{n}'],rowe_stats,ranges)
        #Save tau stats in a h5 file
        self.save_tau_stats(tau_stats, p_bestfits)

        # Save tau plots
        self.tau_plots(tau_stats)

    def load_rowe(self,s):
        f = self.open_input("rowe_stats")   
        rowe_stats = {}

        for i in 0, 1, 2, 3, 4, 5:
            name    = STAR_TYPE_NAMES[s]
            theta   = f['rowe_statistics'][f"rowe_{i}_{name}"]['theta'][:]
            xi_plus = f['rowe_statistics'][f"rowe_{i}_{name}"]['xi_plus'][:]
            xi_minus = f['rowe_statistics'][f"rowe_{i}_{name}"]['xi_minus'][:]
            xip_err  = f['rowe_statistics'][f"rowe_{i}_{name}"]['xip_err'][:]
            xim_err  = f['rowe_statistics'][f"rowe_{i}_{name}"]['xim_err'][:]
            
            rowe_stats[i] = theta, xi_plus, xi_minus, xip_err, xim_err
        
        return rowe_stats

    def sample(self, tau_stats, rowe_stats, ranges, nwalkers=100, ndim=3):
        '''
        Run a simple mcmc chain to detemine the best-fit values for alpha, beta, eta  
        '''
        import emcee
        import scipy.optimize as optimize
 
        _, _, _, _, _, _, _, cov = tau_stats
        mask   = cov.diagonal() > 0
        cov    = cov[mask][:, mask]
        # We debias the covariance with two correction factors from Hartlap et. al 2006 eq. 17 and Dodelson & Schneider 2013 eq. 28
        Njk    = self.config['npatch']
        Ndv    = 6*self.config['nbins']
        if not Ndv < Njk - 2:
            raise ValueError("Hartlap correction only valid for N data vector elements < N jackknife patches - 2")
        f_H    = 1.*(Njk-Ndv-2)/(Njk-1)
        f_DS   = 1/(1+(Ndv-ndim)*(Njk-Ndv-2)/(Njk-Ndv-1)/(Njk-Ndv-4))
        cov    = cov / f_H / f_DS 
        
        _, tau0p, tau0m, tau2p, tau2m, tau5p, tau5m, _  = tau_stats
        invcov = np.linalg.inv(cov)
        
        initguess = [0,-1,1]
        bestpars = optimize.minimize(self.chi2, initguess, args=(tau_stats, rowe_stats, invcov, mask),
                                      method='Nelder-Mead', tol=1e-6)
        initpos = [bestpars.x + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        print("Computing best-fit alpha, beta, eta")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logProb, args=(tau_stats, rowe_stats, ranges, invcov, mask))
        sampler.run_mcmc(initpos, nsteps=5000, progress=False);
        flat_samples = sampler.get_chain(discard=2000,flat=True)
        
        ret = {}
        var = ['alpha','beta','eta']
        for i,v in enumerate(var):
            mcmc   = np.percentile(flat_samples[:, i], [16, 50, 84])
            q      = np.diff(mcmc)
            ret[v] = {'median': mcmc[1],'lerr': q[0], 'rerr': q[1]}
            
        chi2 = self.chi2([ret['alpha']['median'],ret['beta']['median'],ret['eta']['median']], tau_stats,rowe_stats,invcov,mask)
        # degree of freedom = nbins * 6 (i.e. tau 0+/-, tau2+/-, tau5+/-) - ndim
        dof  = (self.config['nbins']*6) - ndim
        print("Best-fit finished. Resulting chi^2/dof: ", chi2/dof)
        return ret

    def logPrior(self,theta,ranges):
        '''
        If parameter in defined range return 0, otherwise -np.inf
        '''
        alpha, beta, eta = theta

        if (ranges['alpha'][0] < alpha < ranges['alpha'][1]) and (ranges['beta'][0] < beta < ranges['beta'][1]) and (ranges['eta'][0] < eta < ranges['eta'][1]):
            return 0.0
        return -np.inf


    def chi2(self, theta, tau_stats, rowe_stats, invcov, mask):
        '''
        Compute likelihood
        theta     : parameters
        tau_stats : Measured tau stats
        rowe_stats: Measured rowe stats to be used for Tau 
        invcov    : Inverse covariance matrix 
        '''
        
        # Load parameters
        alpha, beta, eta = theta

        # Load rowe and tau
        _, rowe0p, rowe0m, _, _  = rowe_stats[0]
        _, rowe1p, rowe1m, _, _  = rowe_stats[1]
        _, rowe2p, rowe2m, _, _  = rowe_stats[2]
        _, rowe3p, rowe3m, _, _  = rowe_stats[3]
        _, rowe4p, rowe4m, _, _  = rowe_stats[4]
        _, rowe5p, rowe5m, _, _  = rowe_stats[5]
        _, tau0p, tau0m, tau2p, tau2m, tau5p, tau5m, _  = tau_stats

        # Create combined template
        T0p    = alpha*rowe0p + beta*rowe2p + eta*rowe5p
        T0m    = alpha*rowe0m + beta*rowe2m + eta*rowe5m
        T2p    = alpha*rowe2p + beta*rowe1p + eta*rowe4p
        T2m    = alpha*rowe2m + beta*rowe1m + eta*rowe4m
        T5p    = alpha*rowe5p + beta*rowe4p + eta*rowe3p
        T5m    = alpha*rowe5m + beta*rowe4m + eta*rowe3m
        
        # Create data and template vector
        Tall  = np.concatenate([T0p, T0m, T2p, T2m, T5p, T5m])[mask]
        Xall  = np.concatenate([tau0p, tau0m, tau2p, tau2m, tau5p, tau5m])[mask]

        return np.dot(Xall-Tall,np.dot(Xall-Tall,invcov))

    def logProb(self, theta, tau_stats, rowe_stats, ranges, invcov, mask):
        lp = self.logPrior(theta,ranges)
        if not np.isfinite(lp):
            return -np.inf
        return lp + (-0.5)*self.chi2(theta, tau_stats, rowe_stats, invcov, mask)


    def compute_all_tau(self, gra, gdec, g, gw, s, sra, sdec, e_meas, e_mod, de, T_f, star_type):
        '''
        Compute tau0, tau2, tau5.
        All three needs to be computed at once due to covariance.

        gra    : RA of galaxies
        gdec   : DEC of galaxies
        g      : shear for observed galaxies np.array((e1, e2)
        gw     : weights

        s      : indices of stars to use in calculation
        sra    : RA of stars
        sdec   : DEC of stars
        
        e_meas : measured ellipticities of PSF from stars -- np.array((e1meas, e2meas))
        e_mod  : model ellipticities of PSF               -- np.array((e1mod, e2mod))
        de     : e_meas-e_mod                              -- np.array((de1, de2))
        T_f    : (T_meas - T_model)/T_meas                -- np.array(T_f)
        '''
        
        import treecorr

        p = e_mod
        q = de
        w = e_meas * T_f
        
        sra, sdec = np.array((sra[star_type==s], sdec[star_type==s])) # Get ra/dec for specific stars
        p = np.array(( [p[0][star_type==s], p[1][star_type==s]]))     # Get p for specific stars
        q = np.array(( [q[0][star_type==s], q[1][star_type==s]]))     # Get q for specific stars
        w = np.array(( [w[0][star_type==s], w[1][star_type==s]]))     # Get w for specific stars

        print(f"Computing Tau 0,2,5 and the covariance")
        self.config["num_threads"] = int(os.environ.get("OMP_NUM_THREADS", 1))
        
        # Load all catalogs
        catg = treecorr.Catalog(ra=gra, dec=gdec, g1=g[0], g2=g[1], w=gw, ra_units="deg", dec_units="deg",npatch=self.config['npatch']) # galaxy shear
        catp = treecorr.Catalog(ra=sra, dec=sdec, g1=p[0], g2=p[1], ra_units="deg", dec_units="deg",patch_centers=catg.patch_centers) # e_model
        catq = treecorr.Catalog(ra=sra, dec=sdec, g1=q[0], g2=q[1], ra_units="deg", dec_units="deg",patch_centers=catg.patch_centers) # (e_* - e_model)
        catw = treecorr.Catalog(ra=sra, dec=sdec, g1=w[0], g2=w[1], ra_units="deg", dec_units="deg",patch_centers=catg.patch_centers) # (e_*(T_* - T_model)/T_* )
        
        # Compute all corrleations
        print(f"Computing shear-e_model correlation {catg.nobj} x {catp.nobj}")
        corr0 = treecorr.GGCorrelation(self.config)
        corr0.process(catg, catp)
        print(f"Computing shear-(e_*-e_model) correlation {catg.nobj} x {catq.nobj}")
        corr2 = treecorr.GGCorrelation(self.config)
        corr2.process(catg, catq)
        print(f"Computing shear-(e_*(T_* - T_model)/T_*) correlation {catg.nobj} x {catw.nobj}")
        corr5 = treecorr.GGCorrelation(self.config)
        corr5.process(catg, catw)
        
        # Estimate covariance using bootstrap. The ordering is xip0,xim0,xip2,xim2,xip5,xim5.
        cov = treecorr.estimate_multi_cov([corr0,corr2,corr5], self.config.cov_method)
        
        return corr0.meanr, corr0.xip, corr0.xim, corr2.xip, corr2.xim, corr5.xip, corr5.xim, cov
        

    def save_tau_stats(self, tau_stats, p_bestfits):
        '''
        tau_stats: (dict) dictionary containing theta,tau0,tau2,tau5 and cov
        '''            
        f = self.open_output("tau_stats")
        g = f.create_group("tau_statistics")

        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
            if self.config['tomographic']:
                with self.open_input("binned_shear_catalog") as h:
                        nbin = list(range(h["shear"].attrs["nbin_source"])) + ['all']
            else:
                nbin = ['all']
            for n in nbin:
                theta, tau0p, tau0m, tau2p, tau2m, tau5p, tau5m, cov = tau_stats[s][f'bin_{n}']
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"tau_{name}/bin_{n}/")
                h.create_dataset("theta" , data=theta)
                h.create_dataset("tau0p" , data=tau0p)
                h.create_dataset("tau0m" , data=tau0m)
                h.create_dataset("tau2p" , data=tau2p)
                h.create_dataset("tau2m" , data=tau2m)
                h.create_dataset("tau5p" , data=tau5p)
                h.create_dataset("tau5m" , data=tau5m)
                h.create_dataset("cov"   , data=cov)

                # Also save best-fit values 
                h = g.create_group(f"bestfits_{name}/bin_{n}")
                alpha_err = max(p_bestfits[s][f'bin_{n}']['alpha']['lerr'],p_bestfits[s][f'bin_{n}']['alpha']['rerr']) 
                beta_err  = max(p_bestfits[s][f'bin_{n}']['beta']['lerr'] ,p_bestfits[s][f'bin_{n}']['beta']['rerr']) 
                eta_err   = max(p_bestfits[s][f'bin_{n}']['eta']['lerr']  ,p_bestfits[s][f'bin_{n}']['eta']['rerr']) 
                h.create_dataset("alpha"    , data=p_bestfits[s][f'bin_{n}']['alpha']['median'])
                h.create_dataset("alpha_err", data=alpha_err)
                h.create_dataset("beta"     , data=p_bestfits[s][f'bin_{n}']['beta']['median'])
                h.create_dataset("beta_err" , data=beta_err)
                h.create_dataset("eta"      , data=p_bestfits[s][f'bin_{n}']['eta']['median'])
                h.create_dataset("eta_err"  , data=eta_err)

        f.close()

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g      = f["stars"]
            ra     = g["ra"][:]
            dec    = g["dec"][:]
            e1meas  = g["measured_e1"][:]
            e2meas  = g["measured_e2"][:]
            e1mod  = g["model_e1"][:]
            e2mod  = g["model_e2"][:]
            if self.config["flip_g2"]:
                e2meas *= -1
                e2mod  *= -1
            de1    = e1meas - e1mod
            de2    = e2meas - e2mod
            
            if self.config["psf_size_units"] == "Tmeas":
                T_frac = (g["measured_T"][:] - g["model_T"][:]) / g["measured_T"][:]
            elif self.config["psf_size_units"] == "Tmodel":
                T_frac = (g["measured_T"][:] - g["model_T"][:]) / g["model_T"][:]  
            elif self.config["psf_size_units"] == "sigma":
                T_frac = (g["measured_T"][:] ** 2 - g["model_T"][:] ** 2) / g["measured_T"][:] ** 2
            else:
                raise ValueError("Need to specify measured_T: Tmeas/Tmodel/sigma")

            if self.config['subtract_mean']:
                e_meas = np.array((e1meas-np.mean(e1meas), e2meas-np.mean(e2meas)))
                e_mod  = np.array((e1mod-np.mean(e1mod)  , e2mod-np.mean(e2mod)))
                de     = np.array((de1-np.mean(de1)      , de2-np.mean(de2)))

            else:
                e_meas = np.array((e1meas, e2meas ))
                e_mod  = np.array((e1mod , e2mod  ))
                de     = np.array((de1   , de2    ))

            star_type = load_star_type(g)

        return ra, dec, e_meas, e_mod, de, T_frac, star_type
    
    def load_galaxies(self, zbin):
        with self.open_input("binned_shear_catalog") as f:
            g   = f[f'shear/bin_{zbin}/']
            ra  = g['ra'][:]
            dec = g['dec'][:]
            g1  = g['g1'][:]
            g2  = g['g2'][:]
            w   = g['weight'][:]
        
        if self.config['subtract_mean']:
            g1 = g1 - np.average(g1,weights=w)
            g2 = g2 - np.average(g1,weights=w)
            
        if self.config['flip_g2']:
            g2 *= -1
            
        return ra, dec, g1, g2, w

    def tau_plots(self, tau_stats):
        # Plot non-tomographic stats 1,3,4
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans
        
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
                
            theta, tau0p, tau0m, tau2p, tau2m, tau5p, tau5m, cov = tau_stats[s]['bin_all']
            nb    = len(theta)
            taus  = {'0p':tau0p, '0m':tau0m, '2p':tau2p, '2m':tau2m, '5p':tau5p, '5m':tau5m}
            errs  = {'0p': np.diag(cov[ 0*nb:1*nb , 0*nb:1*nb ])**0.5,
                     '0m': np.diag(cov[ 1*nb:2*nb , 1*nb:2*nb ])**0.5,
                     '2p': np.diag(cov[ 2*nb:3*nb , 2*nb:3*nb ])**0.5,
                     '2m': np.diag(cov[ 3*nb:4*nb , 3*nb:4*nb ])**0.5,
                     '5p': np.diag(cov[ 4*nb:5*nb , 4*nb:5*nb ])**0.5,
                     '5m': np.diag(cov[ 5*nb:6*nb , 5*nb:6*nb ])**0.5
                    }
    
            for j,i in enumerate([0,2,5]):
                f = self.open_output(f"tau{i}",wrapper=True,figsize=(10,6*len(STAR_TYPES)))
                ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
        
                tr = mtrans.offset_copy(
                                        ax.transData, f.file, 0.05 * (j - 1), 0, units="inches"
                                       )
                plt.errorbar(
                             theta,
                             taus[f'{i}p'],
                             errs[f'{i}p'],
                             fmt=".",
                             label=rf"$\tau_{i}+$",
                             capsize=3,
                             color="blue",
                             transform=tr,
                            )
                plt.errorbar(
                             theta,
                             taus[f'{i}m'],
                             errs[f'{i}m'],
                             fmt=".",
                             label=rf"$\tau_{i}-$",
                             capsize=3,
                             color="red",
                             transform=tr,
                            )
                
                plt.xscale("log")
                plt.yscale("symlog")
                plt.xlabel(r"$\theta$")
                plt.ylabel(rf"$\tau_{i}(\theta)$")
                plt.legend()
                plt.title('Non-tomographic'+STAR_TYPE_NAMES[s])

                f.close()

class TXRoweStatistics(PipelineStage):
    """
    Compute and plot PSF Rowe statistics

    People sometimes think that these statistics are called the Rho statistics,
    because we usually use that letter for them.  Not so.  They are named after
    the wonderful Barney Rowe, now sadly lost to high finance,
    who presented the first two of them in MNRAS 404, 350 (2010).
    """

    name = "TXRoweStatistics"
    parallel = False
    inputs = [("star_catalog", HDFFile),
             ("patch_centers", TextFile)]
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
        "definition"    : 'des-y1',
        "subtract_mean" : False,
        "star_type": 'PSF-reserved',
        "var_method": 'bootstrap',
        "flip_g2"   : False
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")
        self.config["num_threads"] = int(os.environ.get("OMP_NUM_THREADS", 1))
        ra, dec, e_meas, e_mod, de, T_f, star_type = self.load_stars()
        rowe_stats = {}
        for t in STAR_TYPES:
            s = np.where(star_type==t)[0]
            if len(s)==0:
                continue
            if self.config['definition']=='des-y1' or self.config['definition']=='des-y3':
                print("Using DES's definition of Rowes")
                rowe_stats[0, t] = self.compute_rowe(0, s, ra, dec, e_mod, e_mod)
                rowe_stats[1, t] = self.compute_rowe(1, s, ra, dec, de, de)
                rowe_stats[2, t] = self.compute_rowe(2, s, ra, dec, de, e_mod)
                rowe_stats[3, t] = self.compute_rowe(3, s, ra, dec, e_meas * T_f, e_meas * T_f)
                rowe_stats[4, t] = self.compute_rowe(4, s, ra, dec, de, e_meas * T_f)
                rowe_stats[5, t] = self.compute_rowe(5, s, ra, dec, e_mod, e_meas * T_f)
            elif self.config['definition'] == 'hsc-y1' or  self.config['definition'] == 'hsc-y3':
                print("Using HSC's definition of Rowes")
                # de =  g_meas - g_model
                # dT = (T_meas - T_model)/Tmodel
                # rho1 = de x de
                # rho2 = e_mod x de
                # rho3 = e_mod(dT/T_mod) x e_mod(dT/T_mod)
                # rho4 = de(e_mod*dT/T_mod)
                # rho5 = e_mod*(e_mod*dT/T_mod)
                rowe_stats[0, t] = self.compute_rowe(0, s, ra, dec, e_mod, e_mod)
                rowe_stats[1, t] = self.compute_rowe(1, s, ra, dec, de, de)
                rowe_stats[2, t] = self.compute_rowe(2, s, ra, dec, de, e_mod)
                rowe_stats[3, t] = self.compute_rowe(3, s, ra, dec, e_mod * T_f, e_mod * T_f)
                rowe_stats[4, t] = self.compute_rowe(4, s, ra, dec, de, e_mod * T_f)
                rowe_stats[5, t] = self.compute_rowe(5, s, ra, dec, e_mod, e_meas * T_f)
        self.save_stats(rowe_stats)
        self.rowe_plots(rowe_stats)

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g      = f["stars"]
            ra     = g["ra"][:]
            dec    = g["dec"][:]
            e1meas = g["measured_e1"][:]
            e2meas = g["measured_e2"][:]
            e1mod  = g["model_e1"][:]
            e2mod  = g["model_e2"][:]
            if self.config["flip_g2"]:
                e2meas *= -1
                e2mod  *= -1
            de1    = e1meas - e1mod
            de2    = e2meas - e2mod
            if self.config["psf_size_units"] == "Tmeas":
                T_frac = (g["measured_T"][:] - g["model_T"][:]) / g["measured_T"][:]
            elif self.config["psf_size_units"] == "Tmodel":
                T_frac = (g["measured_T"][:] - g["model_T"][:]) / g["model_T"][:]    
            elif self.config["psf_size_units"] == "sigma":
                T_frac = (g["measured_T"][:] ** 2 - g["model_T"][:] ** 2) / g["measured_T"][:] ** 2
            else:
                sys.exit("Need to specify measured_T: Tmeas/Tmodel/sigma")

            if self.config['subtract_mean']:
                e_meas = np.array((e1meas-np.mean(e1meas), e2meas-np.mean(e2meas)))
                e_mod  = np.array((e1mod-np.mean(e1mod)  , e2mod-np.mean(e2mod)))
                de     = np.array((de1-np.mean(de1)      , de2-np.mean(de2)))

            else:
                e_meas = np.array((e1meas, e2meas ))
                e_mod  = np.array((e1mod , e2mod  ))
                de     = np.array((de1   , de2    ))

            star_type = load_star_type(g)

        return ra, dec, e_meas, e_mod, de, T_frac, star_type

    def compute_rowe(self, i, s, ra, dec, q1, q2):
        # select a subset of the stars
        ra  = ra[s]
        dec = dec[s]
        q1  = q1[:, s]
        q2  = q2[:, s]
        n   = len(ra)
        print(f"Computing Rowe statistic rho_{i} from {n} objects")
        import treecorr
    
        corr = treecorr.GGCorrelation(self.config)
        cat1 = treecorr.Catalog(
            ra=ra, dec=dec, g1=q1[0], g2=q1[1], ra_units="deg", dec_units="deg",
            patch_centers=self.get_input("patch_centers")
        )
        cat2 = treecorr.Catalog(
            ra=ra, dec=dec, g1=q2[0], g2=q2[1], ra_units="deg", dec_units="deg",
            patch_centers=self.get_input("patch_centers")
        )
        corr.process(cat1, cat2)
        return corr.meanr, corr.xip, corr.xim, corr.varxip**0.5, corr.varxim**0.5

    def rowe_plots(self, rowe_stats):
        # First plot - stats 1,3,4
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans
        
        f = self.open_output("rowe0",wrapper=True,figsize=(10,6*len(STAR_TYPES)))
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            
            for j,i in enumerate([0]):
                theta, xip, xim, xip_err, xim_err = rowe_stats[i,s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * (j - 1), 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    xip,
                    xip_err,
                    fmt=".",
                    label=rf"$\rho_{i}+$",
                    capsize=3,
                    color="blue",
                    transform=tr,
                )
                plt.errorbar(
                    theta,
                    xim,
                    xim_err,
                    fmt=".",
                    label=rf"$\rho_{i}-$",
                    capsize=3,
                    color="blue",
                    transform=tr,
                )
            plt.xscale("log")
            plt.yscale("symlog")
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$\xi(\theta)$")
            plt.legend()
            plt.title(STAR_TYPE_NAMES[s])
        f.close()

        f = self.open_output("rowe134", wrapper=True, figsize=(10, 6 * len(STAR_TYPES)))
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            mkr = ["o","s","D"]
            for j, i in enumerate([1, 3, 4]):
                theta, xip, xim, xip_err, xim_err = rowe_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * (j - 1), 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    xip,
                    xip_err,
                    marker=mkr[j],
                    label=rf"$\rho_{i}+$",
                    capsize=3,
                    color="blue",
                    transform=tr,
                )
                plt.errorbar(
                    theta,
                    xim,
                    xim_err,
                    marker=mkr[j],
                    label=rf"$\rho_{i}-$",
                    capsize=3,
                    color="red",
                    transform=tr,
                )
            plt.xscale("log")
            plt.yscale("symlog")
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$\xi(\theta)$")
            plt.legend()
            plt.title(STAR_TYPE_NAMES[s])
        f.close()

        f = self.open_output("rowe25", wrapper=True, figsize=(10, 6 * len(STAR_TYPES)))
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            mkr = ["o","s"]
            for j, i in enumerate([2, 5]): 
                theta, xip, xim, xip_err, xim_err = rowe_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * j - 0.025, 0, units="inches"
                )
                plt.errorbar(
                    theta,
                    xip,
                    xip_err,
                    fmt=mkr[j],
                    label=rf"$\rho_{i}+$",
                    capsize=3,
                    color="blue",
                    transform=tr,
                )
                plt.errorbar(
                    theta,
                    xim,
                    xim_err,
                    fmt=mkr[j],
                    label=rf"$\rho_{i}-$",
                    capsize=3,
                    color="red",
                    transform=tr,
                )
                plt.title(STAR_TYPE_NAMES[s])
                plt.xscale("log")
                plt.yscale("symlog")
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$\xi(\theta)$")
                plt.legend()
        f.close()

    def save_stats(self, rowe_stats):
        f = self.open_output("rowe_stats")
        g = f.create_group("rowe_statistics")
        for i in 0, 1, 2, 3, 4, 5:
            for s in STAR_TYPES:
                if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
                theta, xip, xim, xip_err, xim_err = rowe_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"rowe_{i}_{name}")
                h.create_dataset("theta", data=theta)
                h.create_dataset("xi_plus", data=xip)
                h.create_dataset("xi_minus", data=xim)
                h.create_dataset("xip_err", data=xip_err)
                h.create_dataset("xim_err", data=xim_err)
        f.close()

#########################################################################################################################
class TXPHStatistics(PipelineStage):
    """
    Compute and plot PSF Statistics as described in Paulin-Henricksson et. al 2008 
    Heavily drawing upon work by B. Giblin here:
    https://github.com/KiDS-WL/Cat_to_Obs_K1000_P1/tree/master/PSF_systests
    """
    name     = "TXPHStatistics"
    parallel = False
    inputs   = [("binned_shear_catalog"    , ShearCatalog),
                ("star_catalog"            , HDFFile),
                ("patch_centers"           , TextFile),
                ("twopoint_theory_real"    , SACCFile),
               ]

    outputs  = [
                ("PHstat_plus"   , PNGFile),
                ("PHstat_min"    , PNGFile),
                ("ph_stats"       , HDFFile),
               ]


    config_options = {
                       "shear_prefix"  : "",
                       "min_sep"       : 0.5,
                       "max_sep"       : 250.0,
                       "nbins"         : 20,
                       "bin_slop"      : 0.01,
                       "sep_units"     : "arcmin",
                       "npatch"        : 150,
                       "psf_size_units": "sigma",
                       "subtract_mean" : False,
                       "dec_cut"       : True,           # affects KiDS-1000 only
                       "star_type"     : 'PSF-reserved',
                       "cov_method"    : 'bootstrap',
                       "flip_g2"       : False,
                       "ang_cell"      : 5./60.,
                       "nboot"         : 30,
                     }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")

        # Load star properties
        sra, sdec, e_psf, de, T_psf, dT, star_type = self.load_stars()

        # Compute size quantities
        dT_Tg_ratio, Tg_invsq, dT_Tg_ratio_tot, Tg_invsq_tot, num_zbin, nzbin_tot = self.compute_Tquantities(sra,sdec,dT)

        
        # Compute PH correlations
        ph_corr, ph_stats = {}, {}
        for t in STAR_TYPES:
            s = np.where(star_type==t)[0]
            if len(s)==0:
                continue
            ph_corr[0, t] = self.compute_PHCorr(0, s, sra, sdec, e_psf, e_psf, dT, dT)
            ph_corr[1, t] = self.compute_PHCorr(1, s, sra, sdec, e_psf, de, dT, T_psf)
            ph_corr[2, t] = self.compute_PHCorr(2, s, sra, sdec, de, de, T_psf, T_psf)
        
            ph_stats[t] = self.compute_PHStat(ph_corr, dT_Tg_ratio, Tg_invsq, dT_Tg_ratio_tot,
                                              Tg_invsq_tot, num_zbin, nzbin_tot)

        self.save_stats(ph_stats)
        self.ph_plots(ph_stats, num_zbin)
        
    def compute_Tquantities(self, sra, sdec, dT):
        with self.open_input("binned_shear_catalog") as f:
            num_zbin = f["shear"].attrs["nbin_source"]

         # Compute mean dT/T_g, 1/T_g**2 & errors for each redshift bin
        dT_Tg_ratio = np.zeros([2, num_zbin])      
        Tg_invsq = np.zeros_like(dT_Tg_ratio)       

        # RA values that cross 0 causes issues with interpolation. Convert.
        sra[((sra<360) & (sra>300))] += -360.
        
        # Load galaxy properties
        for n in range(num_zbin):
            gra, gdec, Tgal, T_psf_g, wg = self.load_galaxies(n)
        
            #convert RA values.
            gra[((gra<360) & (gra>300))] += -360. 
            
            # Compute dT at the position of the galaxy
            dT_interpg = self.interpolate_dT(sra, sdec, dT,gra, gdec)

            dT_Tg_ratio[0,n] = np.average(dT_interpg/Tgal, weights=wg)
            dT_Tg_ratio[1,n] = self.estimate_Terror(self.config["nboot"],
                                                    dT_interpg/Tgal, weights=wg)

            Tg_inv = np.average(1./Tgal, weights=wg)
            Tg_invsq[0,n] = Tg_inv**2
            Tg_invsq[1,n] = np.sqrt(2 * Tg_inv) * self.estimate_Terror(self.config["nboot"],
                                                                       1/Tgal, weights=wg)

        # For cross-bins, average the Tquantities from the individual bins
        nzbin_tot = np.sum(range(num_zbin+1))
        dT_Tg_ratio_tot = np.zeros([2, nzbin_tot])
        Tg_invsq_tot = np.zeros_like(dT_Tg_ratio_tot )

        k=0
        for i in range(num_zbin):
            for j in range(num_zbin):
                if j>= i:
                    dT_Tg_ratio_tot[0,k] = (dT_Tg_ratio[0,i] + dT_Tg_ratio[0,j]) / 2
                    dT_Tg_ratio_tot[1,k] = np.sqrt(dT_Tg_ratio[1,i]**2 + dT_Tg_ratio[1,j]**2)
                    Tg_invsq_tot[0,k]    = (Tg_invsq[0,i] + Tg_invsq[0,j]) / 2
                    Tg_invsq_tot[1,k]    = np.sqrt(Tg_invsq[0,i]**2 + Tg_invsq[0,j]**2)
                    k+=1
                    
        return dT_Tg_ratio, Tg_invsq, dT_Tg_ratio_tot, Tg_invsq_tot, num_zbin, nzbin_tot
        
    def interpolate_dT(self, sra, sdec, dT, gra, gdec):
        from scipy.stats import binned_statistic_2d

        # We need to make a grid for interpolation 
        # Load angular size for 1 cell (default 5 arcmin) & make bins for the grid
        ang_cell = self.config["ang_cell"]
        nbins_x = int((sra.max() - sra.min()) / ang_cell)
        nbins_y = int((sdec.max() - sdec.min()) / ang_cell)
        
        # Compute cell coordinates of stars                                                                        
        X_p = nbins_x * (sra - sra.min()) / (sra.max()-sra.min())
        Y_p = nbins_y * (sdec - sdec.min()) / (sdec.max()-sdec.min())
        
        # Make dT grid:                                                                                
        sum_wdT_grid, _, _, _ = binned_statistic_2d(Y_p, X_p, dT * np.ones_like(dT),
                                                    statistic='sum', bins=[nbins_y, nbins_x])
        sum_grid, _, _, _    = binned_statistic_2d(Y_p, X_p, np.ones_like(dT),
                                                   statistic='sum', bins=[nbins_y, nbins_x])
        Av_grid = sum_wdT_grid / sum_grid
        dT_grid = np.nan_to_num(Av_grid, nan=0.)
        
        # Append last row, column to fix interpolation error:                                                      
        dT_grid = np.c_[dT_grid, dT_grid[:,-1]]
        dT_grid = np.r_[dT_grid, [dT_grid[-1,:]]]
        
        # Get cell coordinates of galaxies, round down
        X_g = nbins_x * (gra - sra.min()) / (sra.max()-sra.min())
        Y_g = nbins_y * (gdec - sdec.min()) / (sdec.max()-sdec.min())
        Xi = X_g.astype(int)
        Yi = Y_g.astype(int)                                                       
        
        # Calculate dT @ the galaxy's position
        VAL_XYlo = dT_grid[Yi, Xi]    + (X_g - Xi) * (dT_grid[Yi, Xi + 1]    - dT_grid[Yi, Xi])
        VAL_XYhi = dT_grid[Yi + 1,Xi] + (X_g - Xi) * (dT_grid[Yi + 1,Xi + 1] - dT_grid[Yi+1, Xi])
        dT_interpg = VAL_XYlo + (Y_g - Yi) * (VAL_XYhi - VAL_XYlo)
        
        return dT_interpg
        
    def estimate_Terror(self, nboot, Tquant, weights):
        # Bootstrapped error estimation for size quantity
        N = len(Tquant)
        samples = np.zeros(nboot)       
        for i in range(nboot):
            idx = np.random.randint(0,N,N)                         
            samples[i] = np.sum(weights[idx] * Tquant[idx]) / np.sum(weights[idx])
            
        return np.std(samples)
    
    def compute_PHCorr(self, i, s, ra, dec, q1, q2, q3, q4):
        import treecorr
        
        # Select a subset of the stars
        ra, dec, n = ra[s], dec[s], len(ra)
        print(np.shape(q1),np.shape(q2),np.shape(q3),np.shape(q4))
        q1 = q1[:, s]
        q2 = q2[:, s]
        q3 = q3[s]
        q4 = q4[s]
        
        print(f"Computing PH statistic ph_{i} from {n} objects")
        corr = treecorr.GGCorrelation(self.config)
        
        cat1 = treecorr.Catalog(
            ra = ra, dec = dec, g1 = q1[0] * q3,g2 = q1[1] * q3, ra_units="deg", dec_units="deg",
            patch_centers = self.get_input("patch_centers"))
        
        cat2 = treecorr.Catalog(
            ra = ra, dec = dec, g1 = q2[0] * q4, g2 = q2[1]* q4, ra_units="deg", dec_units="deg",
            patch_centers = self.get_input("patch_centers"))
        corr.process(cat1, cat2)
        
        return corr.meanr, corr.xip, corr.xim, corr.varxip**0.5, corr.varxim**0.5

    def compute_PHStat(self, phcorr, dT_Tg_ratio, Tg_invsq, dT_Tg_ratio_tot, Tg_invsq_tot, num_zbin, nzbin_tot):
        # Finally combine theory data vector, correlations, and T quantities
        # to create \delta\xi as shown in Eq.10 of Giblin et al. 2021 
        
        dxip = np.zeros([nzbin_tot, self.config["nbins"]])
        dxim = np.zeros_like(dxip)
        err_dxip = np.zeros_like(dxip)
        err_dxim = np.zeros_like(dxip)
        
        dxip_terms = np.zeros([nzbin_tot, self.config["nbins"], 4])
        dxim_terms = np.zeros_like(dxip_terms)
        err_dxip_terms = np.zeros_like(dxip_terms)
        err_dxim_terms = np.zeros_like(dxip_terms)
        
        for s in STAR_TYPES:
                if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
                    
        # Load theory data vector
        ttht, txip, txim = self.load_theory(num_zbin,nzbin_tot)
        
        for n in range(nzbin_tot):
                # Also interpolate the theory vector onto the theta bins of the PH-stats                
                dxip_terms[n,:,0] = 2 * np.interp(phcorr[0, s][0], ttht[n], txip[n]) * dT_Tg_ratio_tot[0,n]
                dxim_terms[n,:,0] = 2 * np.interp(phcorr[0, s][0], ttht[n], txim[n]) * dT_Tg_ratio_tot[0,n]
                dxip_terms[n,:,1] =     Tg_invsq_tot[0,n] * (phcorr[0, s][1])
                dxim_terms[n,:,1] =     Tg_invsq_tot[0,n] * (phcorr[0, s][2])
                dxip_terms[n,:,2] = 2 * Tg_invsq_tot[0,n] * (phcorr[1, s][1])
                dxim_terms[n,:,2] = 2 * Tg_invsq_tot[0,n] * (phcorr[1, s][2])
                dxip_terms[n,:,3] =     Tg_invsq_tot[0,n] * (phcorr[2, s][1])
                dxim_terms[n,:,3] =     Tg_invsq_tot[0,n] * (phcorr[2, s][2])
                
                # Total
                dxip[n,:] = dxip_terms[n,:,0] + dxip_terms[n,:,1] + dxip_terms[n,:,2] + dxip_terms[n,:,3]
                dxim[n,:] = dxim_terms[n,:,0] + dxim_terms[n,:,1 ]+ dxim_terms[n,:,2] + dxim_terms[n,:,3]
            
                # Propagate Errors
                err_dxip_terms[ n,:,0] = 2*np.interp(phcorr[0, s][0], ttht[n], txip[n]) * dT_Tg_ratio_tot[ 1,j]
                err_dxim_terms[ n,:,0] = 2*np.interp(phcorr[0, s][0], ttht[n], txim[n]) * dT_Tg_ratio_tot[ 1,j]
            
                # PH term 2 has a factor of 2
                scale = [1,2,1] 
                # cycle through 3 ph terms - same error form
                for t in range(1,4): 
                        part1 = scale[t-1] * Tg_invsq_tot[1,n]**2 * phcorr[t-1, s][1]**2  
                        part2 = scale[t-1] * Tg_invsq_tot[0,n]**2 * phcorr[t-1, s][3]**2 
                        err_dxip_terms[ n,:,t] = (part1 + part2)**0.5
                        
                        part1 = scale[t-1] * Tg_invsq_tot[1,n]**2 * phcorr[t-1, s][2]**2  
                        part2 = scale[t-1] * Tg_invsq_tot[0,n]**2 * phcorr[t-1, s][4]**2 
                        err_dxim_terms[n,:,t] = (part1 + part2)**0.5

                err_dxip[n,:] = (  err_dxip_terms[n,:,0]**2 + err_dxip_terms[n,:,1]**2 
                                 + err_dxip_terms[n,:,2]**2 + err_dxip_terms[n,:,3]**2 )**0.5
                err_dxim[n,:] = (  err_dxim_terms[n,:,0]**2 + err_dxim_terms[n,:,1]**2 
                                 + err_dxim_terms[n,:,2]**2 + err_dxim_terms[n,:,3]**2 )**0.5
            
        return phcorr[0, s][0], dxip, dxim, err_dxip, err_dxim, dxip_terms, dxim_terms, err_dxip_terms, err_dxim_terms    

    def load_theory(self,num_zbin,nzbin_tot):
        import sacc
        
        filename_theory = self.get_input("twopoint_theory_real")
        s = sacc.Sacc.load_fits(filename_theory)
        theta = np.zeros((nzbin_tot,9)) #maybe read this in through the sacc info?
        xip, xim = np.zeros_like(theta), np.zeros_like(theta)
        k = 0
        for i in range(num_zbin):
                for j in range(num_zbin):
                    #if j>=i:
                    theta_, xip_ = s.get_theta_xi('galaxy_shear_xi_plus',
                                                  'source_' + f'{i}','source_' + f'{j}')
                    
                    theta_, xim_ = s.get_theta_xi('galaxy_shear_xi_minus',
                                                  'source_' + f'{i}','source_' + f'{j}')
                    if len(xip_) != 0:
                        theta[k], xip[k],xim[k] = theta_, xip_, xim_
                        k+=1
                        
        return theta, xip, xim
                        
    def load_galaxies(self,zbin):
        with self.open_input("binned_shear_catalog") as f:
        # probably need some kind of line here to load prefix    
            g     = f[f"shear/bin_{zbin}/"]
            ra    = g["ra"][:]
            dec   = g["dec"][:]
            Tgal  = g["T"][:]
            Tpsf  = g["psf_T_mean"][:]
            w     = g["weight"][:]
            
        return ra, dec, Tgal, Tpsf, w

    def load_stars(self):
        with self.open_input("star_catalog") as f:
            g      = f["stars"]
            ra     = g["ra"][:]
            dec    = g["dec"][:]
            e1meas = g["measured_e1"][:]
            e2meas = g["measured_e2"][:]
            e1mod  = g["model_e1"][:]
            e2mod  = g["model_e2"][:]
            if self.config["flip_g2"]:
                e2meas *= -1
                e2mod  *= -1
            de1    = e1meas - e1mod
            de2    = e2meas - e2mod
             
            if self.config["psf_size_units"] == "sigma":
                Tmeas  =  g["measured_T"][:] ** 2
                dT     = (g["measured_T"][:] ** 2 - g["model_T"][:] ** 2) / g["measured_T"][:] ** 2
            else:
                Tmeas  =  g["measured_T"][:]
                dT     = (g["measured_T"][:] - g["model_T"][:])

            if self.config['subtract_mean']:
                e_meas = np.array((e1meas - np.mean(e1meas), e2meas - np.mean(e2meas)))
                de     = np.array((de1 - np.mean(de1)      , de2 - np.mean(de2)))

            else:
                e_meas = np.array((e1meas, e2meas ))
                de     = np.array((de1   , de2    ))

            star_type = load_star_type(g)

        return ra, dec, e_meas, de, Tmeas, dT, star_type

    def save_stats(self, ph_stats):
        
        f = self.open_output("ph_stats")
        g = f.create_group("ph_statistics")
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
            name = STAR_TYPE_NAMES[s]
            h = g.create_group(f"ph_{name}")
            h.create_dataset("theta", data=ph_stats[s][0])
            h.create_dataset("deltaxi_plus_sys", data=ph_stats[s][1])
            h.create_dataset("deltaxi_minus_sys", data=ph_stats[s][2])
            h.create_dataset("deltaxi_plus_sys_err", data=ph_stats[s][3])
            h.create_dataset("deltaxi_minus_sys_err", data=ph_stats[s][4])
            
            h.create_dataset("PH_1_plus", data=ph_stats[s][5][:,:,0])
            h.create_dataset("PH_1_minus", data=ph_stats[s][6][:,:,0])
            
            h.create_dataset("PH_1_plus_err", data=ph_stats[s][7][:,:,0])
            h.create_dataset("PH_1_minus_err", data=ph_stats[s][8][:,:,0])
            
            h.create_dataset("PH_2_plus", data=ph_stats[s][5][:,:,1])
            h.create_dataset("PH_2_minus", data=ph_stats[s][6][:,:,1])
            
            h.create_dataset("PH_2_plus_err", data=ph_stats[s][7][:,:,2])
            h.create_dataset("PH_2_minus_err", data=ph_stats[s][8][:,:,2])
            
            h.create_dataset("PH_3_plus", data=ph_stats[s][5][:,:,3])
            h.create_dataset("PH_3_minus", data=ph_stats[s][6][:,:,3])
            
            h.create_dataset("PH_3_plus_err", data=ph_stats[s][7][:,:,3])
            h.create_dataset("PH_3_minus_err", data=ph_stats[s][8][:,:,3])
        f.close()

    def ph_plots(self, ph_stats,num_zbin):
        import matplotlib.pyplot as plt
        # we're only going to produce plots for autocorrelation bin pairs
        k = np.linspace(2,num_zbin-1,num_zbin-2)
        autocorrbins = np.concatenate(([0,num_zbin],k*num_zbin-k*(k-1)/2)
        
        f = self.open_output("PHstat_plus", wrapper=True,
                             figsize=(4 * num_zbin,6 * len(STAR_TYPES)))
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
            _, axes = plt.subplots(num_zbin, len(STAR_TYPES),
                                   squeeze=False, num=f.file.number)
            for j, bn in enumerate(autocorrbins): 
                plt.sca(axes[j, s])
                plt.errorbar(
                    ph_stats[s][0],
                    np.abs(ph_stats[s][1][bn,:]),
                    ph_stats[s][3][bn,:],
                    fmt="o", capsize=3,color="black",
                    label=r"$\delta\xi_+^{\rm sys}$")
                
                plt.plot( ph_stats[s][0], 
                    np.abs(ph_stats[s][5][bn,:,0]),
                    color="xkcd:taupe", linestyle="None", 
                    marker="o",ms=5,
                   label = r"PH$_1$")
        
                plt.plot(ph_stats[s][0], 
                    np.abs(ph_stats[s][5][bn,:,1]),
                    color="rosybrown", linestyle="None", 
                    marker="o", ms=5,
                    label = r"PH$_2$")
            
                plt.plot( ph_stats[s][0],
                    np.abs(ph_stats[s][5][bn,:,2]),
                    color="lightsteelblue", linestyle="None", 
                    marker="o",ms=5,
                    label = r"PH$_3$")
                
                plt.plot( ph_stats[s][0],
                    np.abs(ph_stats[s][5][bn,:,3]),
                    color="slategray", linestyle="None", 
                    marker="o",ms=5,
                    label = r"PH$_4$")
                plt.ylim(bottom=1e-12)
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$|\delta\xi_+^{\rm sys},{\rm terms}|$")
            axes[0, s].set_title(STAR_TYPE_NAMES[s])
            axes[0, 1].legend(bbox_to_anchor=(1.5, 1), loc="upper right")
        f.close()

        f = self.open_output("PHstat_min", wrapper=True,
                             figsize=(4 * num_zbin,6 * len(STAR_TYPES)))
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                continue
            _, axes = plt.subplots(num_zbin, len(STAR_TYPES),
                                   squeeze=False, num=f.file.number)
            for j, bn in enumerate(autocorrbins): 
                plt.sca(axes[j, s])
                plt.errorbar(
                    ph_stats[s][0],
                    np.abs(ph_stats[s][2][bn,:]),
                    ph_stats[s][4][bn,:],
                    fmt="o",capsize=3,color="black",
                    label=r"$\delta\xi_-^{\rm sys}$")
                
                plt.plot( ph_stats[s][0], 
                    np.abs(ph_stats[s][6][bn,:,0]),
                    color="xkcd:taupe", linestyle="None", 
                    marker="o",ms=5,
                    label = r"PH$_1$")
                
                plt.plot(ph_stats[s][0], 
                    np.abs(ph_stats[s][6][bn,:,1]),
                    color="rosybrown", linestyle="None", 
                    marker="o", ms=5,
                    label = r"PH$_2$")
            
                plt.plot( ph_stats[s][0],
                    np.abs(ph_stats[s][6][bn,:,2]),
                    color="lightsteelblue", linestyle="None", 
                    marker="o",ms=5,
                    label = r"PH$_3$")
                
                plt.plot( ph_stats[s][0], 
                    np.abs(ph_stats[s][6][bn,:,3]),
                    color="slategray", linestyle="None", 
                    marker="o",ms=5,
                    label = r"PH$_4$")
                plt.ylim(bottom=1e-12)
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$|\delta\xi_-^{\rm sys},{\rm terms}|$")
            axes[0, s].set_title(STAR_TYPE_NAMES[s])
            axes[0, 1].legend(bbox_to_anchor=(1.5, 1), loc="upper right")
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
        "star_type": 'PSF-reserved',
        "flip_g2": False,
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")
        self.config["num_threads"] = int(os.environ.get("OMP_NUM_THREADS", 1))
        ra, dec, e_psf, de_psf, star_type = self.load_stars()
        ra_gal, dec_gal, g1, g2, weight = self.load_galaxies()

        # only use reserved stars for this statistics
        galaxy_star_stats = {}
        star_star_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            if STAR_TYPE_NAMES[t] != self.config.star_type:
                    continue
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
        _, cal = Calibrator.load(self.get_input("shear_tomography_catalog"))

        # load tomography data
        with self.open_input("shear_tomography_catalog") as f:
            source_bin = f["tomography/bin"][:]
            mask = source_bin != -1  # Only use the sources that pass the fiducial cuts
            if cat_type == "metacal":
                R_total_2d = f["response/R_S_2d"][:] + f["response/R_gamma_mean_2d"][:]
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
            g1, g2 = cal.apply(g1,g2)

        elif cat_type == "lensfit":
            # In KiDS, the additive bias is calculated and removed per North and South field
            # therefore, we add dec to split data into these fields. 
            # You can choose not to by setting dec_cut = 90 in the config, for example.
            g1, g2 = cal.apply(dec,g1,g2)
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
            g1=g1,
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
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            for j, i in enumerate([1, 2]):
                theta, xi, err = galaxy_star_stats[i, s]
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=f"galaxy cross star {TEST_TYPES[i-1]}",
                    capsize=3,
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
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            for j, i in enumerate([1, 2]):
                theta, xi, err = star_star_stats[i, s]
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=f"star cross star {TEST_TYPES[i-1]}",
                    capsize=3,
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
                if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
                theta, xi, err = galaxy_star_stats[i, s]
                name = STAR_TYPE_NAMES[s]
                h = g.create_group(f"star_cross_galaxy_{i}_{name}")
                h.create_dataset("theta", data=theta)
                h.create_dataset("xi_plus", data=xi)
                h.create_dataset("xi_err", data=err)

        g = f.create_group("star_cross_star")
        for i in 1, 2:
            for s in STAR_TYPES:
                if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
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
        "star_type": 'PSF-reserved',
        "flip_g2": False,
    }

    def run(self):
        import treecorr
        import h5py
        import matplotlib

        matplotlib.use("agg")
        self.config["num_threads"] = int(os.environ.get("OMP_NUM_THREADS", 1))
        ra, dec, star_type = self.load_stars()
        ra_gal, dec_gal = self.load_galaxies()
        ra_random, dec_random = self.load_randoms()

        galaxy_star_stats = {}
        for t in STAR_TYPES:
            s = star_type == t
            if STAR_TYPE_NAMES[t] != self.config.star_type:
                    continue
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
            source_bin = f["tomography/bin"][:]
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

        nn.calculateXi(rr=rr, dr=nr, rd=rn)
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

        nn.calculateXi(rr=rr, dr=nr, rd=rn)
        return nn.meanr, nn.xi, nn.varxi**0.5

    def galaxy_star_plots(self, galaxy_star_stats):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtrans

        f = self.open_output(
            "star_density_test", wrapper=True, figsize=(10, 6 * len(STAR_TYPES))
        )
        TEST_TYPES = ["star cross galaxy", "star cross star"]
        for s in STAR_TYPES:
            if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
            ax = plt.subplot(len(STAR_TYPES), 1, s + 1)
            for j, i in enumerate([1, 2]):
                theta, xi, err = galaxy_star_stats[i, s]
                tr = mtrans.offset_copy(
                    ax.transData, f.file, 0.05 * j - 0.025, 0, units="inches")
                plt.errorbar(
                    theta,
                    abs(xi),
                    err,
                    fmt=".",
                    label=f"{TEST_TYPES[i-1]} density stats",
                    capsize=3,
                    # transform=tr,
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
                if STAR_TYPE_NAMES[s] != self.config.star_type:
                    continue
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
    used     = data["calib_psf_used"][:].astype('int')
    reserved = data["calib_psf_reserved"][:].astype('int')
    
    star_type              = np.full(used.size, -99, dtype=int)
    star_type[used==1]     = STAR_PSF_USED
    star_type[reserved==1] = STAR_PSF_RESERVED

    return star_type
