from .base_stage import PipelineStage
from .data_types import Directory, ShearCatalog, HDFFile, PNGFile, TomographyCatalog
from parallel_statistics import ParallelMeanVariance
from .utils.calibration_tools import calculate_selection_response, calculate_shear_response, apply_metacal_response, apply_lensfit_calibration, MeanShearInBins, apply_hsc_calibration
from .utils.fitting import fit_straight_line
from .plotting import manual_step_histogram
import numpy as np

class TXDiagnosticPlots(PipelineStage):
    """
    """
    name='TXDiagnosticPlots'

    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        ('lens_tomography_catalog', TomographyCatalog),
    ]

    outputs = [
        ('g_psf_T', PNGFile),
        ('g_psf_g', PNGFile),
        ('g1_hist', PNGFile),
        ('g2_hist', PNGFile),
        ('g_snr', PNGFile),
        ('g_T', PNGFile),
        ('snr_hist', PNGFile),
        ('mag_hist', PNGFile),
        ('response_hist', PNGFile),

    ]

    config_options = {
        'chunk_rows': 100000,
        'delta_gamma': 0.02,
        'shear_prefix':'mcal_',
        'psf_prefix': 'mcal_psf_',
        'T_min': 0.2,
        'T_max': 0.28,
    }

    def run(self):
        # PSF tests
        import matplotlib
        matplotlib.use('agg')

        with self.open_input('shear_catalog', wrapper=True) as f:
            self.config['shear_catalog_type'] = f.catalog_type

        # Collect together all the methods on this class called self.plot_*
        # They are all expected to be python coroutines - generators that
        # use the yield feature to pause and wait for more input.
        # We instantiate them all here
        
        plotters = [getattr(self, f)() for f in dir(self) if f.startswith('plot_')]

        # Start off each of the plotters.  This will make them all run up to the
        # first yield statement, then pause and wait for the first chunk of data
        for plotter in plotters:
            plotter.send(None)

        # Create an iterator for reading through the input data.
        # This method automatically splits up data among the processes,
        # so the plotters should handle this.
        chunk_rows = self.config['chunk_rows']
        psf_prefix = self.config['psf_prefix']
        shear_prefix = self.config['shear_prefix']
        if self.config['shear_catalog_type']=='metacal':
            shear_cols = [f'{psf_prefix}g1', f'{psf_prefix}g2', f'{psf_prefix}T_mean', 'mcal_g1','mcal_g1_1p','mcal_g1_2p','mcal_g1_1m','mcal_g1_2m','mcal_g2','mcal_g2_1p','mcal_g2_2p','mcal_g2_1m','mcal_g2_2m','mcal_s2n','mcal_T',
                     'mcal_T_1p','mcal_T_2p','mcal_T_1m','mcal_T_2m','mcal_s2n_1p','mcal_s2n_2p','mcal_s2n_1m',
                     'mcal_s2n_2m', 'weight']
        else:
            shear_cols = ['psf_g1','psf_g2','g1','g2','psf_T_mean','s2n','T','weight','m','sigma_e','c1','c2']

        photo_cols = ['mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_y']
        shear_tomo_cols = ['source_bin']
        lens_tomo_cols = ['lens_bin']
        photo_cols = ['mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_y']
        shear_tomo_cols = ['source_bin']
        lens_tomo_cols = ['lens_bin']

        if self.config['shear_catalog_type']=='metacal':
            it = self.combined_iterators(chunk_rows,
                                         'shear_catalog', 'shear', shear_cols,
                                         'photometry_catalog', 'photometry', photo_cols,
                                         'shear_tomography_catalog','tomography',shear_tomo_cols,
                                         'shear_tomography_catalog','metacal_response', ['R_gamma'],
                                         'lens_tomography_catalog','tomography',lens_tomo_cols)
        else:
            it = self.combined_iterators(chunk_rows,
                                         'shear_catalog', 'shear', shear_cols,
                                         'photometry_catalog', 'photometry', photo_cols,
                                         'shear_tomography_catalog','tomography',shear_tomo_cols,
                                         'shear_tomography_catalog','response', ['R'],
                                         'lens_tomography_catalog','tomography',lens_tomo_cols)


        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for (start, end, data) in it:
            print(f"Read data {start} - {end}")
            # This causes each data = yield statement in each plotter to
            # be given this data chunk as the variable data.

            for plotter in plotters:
                plotter.send(data)


        # Tell all the plotters to finish, collect together results from the different
        # processors, and make their final plots.  Plotters need to respond
        # to the None input and
        for plotter in plotters:
            try:
                plotter.send(None)
            except StopIteration:
                pass

    def plot_psf_shear(self):
        # mean shear in bins of PSF
        print("Making PSF shear plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        from .utils.fitting import fit_straight_line
        
        delta_gamma = self.config['delta_gamma']
        size = 5
        gr = np.logspace(-3,-2, size//2)
        psf_g_edges = np.concatenate([-gr[::-1], gr])
        print('psf_g_edges', psf_g_edges)
        psf_prefix = self.config['psf_prefix']

        p1 = MeanShearInBins(f'{psf_prefix}g1', psf_g_edges, delta_gamma, cut_source_bin=True, shear_catalog_type=self.config['shear_catalog_type'])
        p2 = MeanShearInBins(f'{psf_prefix}g2', psf_g_edges, delta_gamma, cut_source_bin=True,shear_catalog_type=self.config['shear_catalog_type'])

        psf_g_mid = 0.5*(psf_g_edges[1:] + psf_g_edges[:-1])

        while True:
            data = yield

            if data is None:
                break

            p1.add_data(data)
            p2.add_data(data)

        mu1, mean11, mean12, std11, std12 = p1.collect(self.comm)
        mu2, mean21, mean22, std21, std22 = p2.collect(self.comm)

        if self.rank != 0:
            return

        fig = self.open_output('g_psf_g', wrapper=True)
        #Include a small shift to be able to see the g1 / g2 points on the plot
        dx = 0.1*(mu1[1] - mu1[0])


        slope11, intercept11, mc_cov = fit_straight_line(mu1, mean11, y_err=std11, nan_error=True, skip_nan=True)
        std_err11 = mc_cov[0,0]**0.5
        line11 = slope11*(mu1)+intercept11

        slope12, intercept12, mc_cov = fit_straight_line(mu1, mean12, y_err=std12, nan_error=True, skip_nan=True)
        std_err12 = mc_cov[0,0]**0.5
        line12 = slope12*(mu1)+intercept12

        slope21, intercept21, mc_cov = fit_straight_line(mu2, mean21, y_err=std21, nan_error=True, skip_nan=True)
        std_err21 = mc_cov[0,0]**0.5
        line21 = slope21*(mu2)+intercept21

        slope22, intercept22, mc_cov = fit_straight_line(mu2, mean22, y_err=std22, nan_error=True, skip_nan=True)
        std_err22 = mc_cov[0,0]**0.5
        line22 = slope22*(mu2)+intercept22

        plt.subplot(2,1,1)
        
        plt.plot(mu1,line11,color='red',label=r"$m=%.4f \pm %.4f$" %(slope11, std_err11))
        plt.plot(mu1,[0]*len(line11),color='black')

        plt.plot(mu1,line12,color='blue',label=r"$m=%.4f \pm %.4f$" %(slope12, std_err12))
        plt.plot(mu1,[0]*len(line12),color='black')
        plt.errorbar(mu1+dx, mean11, std11, label='g1', fmt='+',color='red')
        plt.errorbar(mu1-dx, mean12, std12, label='g2', fmt='+',color='blue')
        plt.xlabel("PSF g1")
        plt.ylabel("Mean g")
        plt.legend()


        plt.subplot(2,1,2)

        plt.plot(mu2,line21,color='red',label=r"$m=%.4f \pm %.4f$" %(slope21, std_err21))
        plt.plot(mu2,[0]*len(line21),color='black')

        plt.plot(mu2,line22,color='blue',label=r"$m=%.4f \pm %.4f$" %(slope22, std_err22))
        plt.plot(mu2,[0]*len(line22),color='black')
        plt.errorbar(mu2+dx, mean21, std21, label='g1', fmt='+',color='red')
        plt.errorbar(mu2-dx, mean22, std22, label='g2', fmt='+',color='blue')
        plt.xlabel("PSF g2")
        plt.ylabel("Mean g")
        plt.legend()
        plt.tight_layout()

        # This also saves the figure
        fig.close()

    def plot_psf_size_shear(self):
        # mean shear in bins of PSF
        print("Making PSF size plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        
        psf_prefix = self.config['psf_prefix']
        
        delta_gamma = self.config['delta_gamma']
        size = 5
        T_min = self.config['T_min']
        T_max = self.config['T_max']
        psf_T_edges = np.linspace(T_min, T_max, size+1)

        binnedShear = MeanShearInBins(f'{psf_prefix}T_mean', psf_T_edges, delta_gamma, cut_source_bin=True, shear_catalog_type=self.config['shear_catalog_type'])
            
        while True:
            data = yield

            if data is None:
                break

            binnedShear.add_data(data)
            

        mu, mean1, mean2, std1, std2 = binnedShear.collect(self.comm)


        if self.rank != 0:
            return

        w = (mu!=0) & np.isfinite(std1)
        mu = mu[w]
        mean1 = mean1[w]
        mean2 = mean2[w]
        std1 = std1[w]
        std2 = std2[w]

        dx = 0.05*(psf_T_edges[1] - psf_T_edges[0])
        slope1, intercept1, cov1 = fit_straight_line(mu, mean1, std1, skip_nan=True, nan_error=True)
        std_err1 = cov1[0,0]**0.5
        line1 = slope1*mu + intercept1
        slope2, intercept2, cov2 = fit_straight_line(mu, mean2, std2, skip_nan=True, nan_error=True)
        std_err2 = cov2[0,0]**0.5
        line2 = slope2*mu + intercept2


        fig = self.open_output('g_psf_T', wrapper=True)

        plt.plot(mu,line1,color='red',label=r"$m=%.4f \pm %.4f$" %(slope1, std_err1))
        plt.plot(mu,[0]*len(mu),color='black')
        plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+',color='red')
        plt.legend(loc='best')

        plt.plot(mu-dx,line2,color='blue',label=r"$m=%.4f \pm %.4f$" %(slope2, std_err2))
        plt.plot(mu,[0]*len(mu),color='black')
        plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+',color='blue')
        plt.xlabel("PSF T")
        plt.ylabel("Mean g")
        #plt.ylim(-0.0015,0.0015)
        plt.legend(loc='best')
        plt.tight_layout()
        fig.close()

    def plot_snr_shear(self):
        # mean shear in bins of snr
        print("Making mean shear SNR plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # Parameters of the binning in SNR
        size = 5
        delta_gamma = self.config['delta_gamma']
        shear_prefix = self.config['shear_prefix']
        snr_edges = np.logspace(.1,2.5,size+1)

        # This class includes all the cutting and calibration, both for 
        # estimator and selection biases
        binnedShear = MeanShearInBins(f'{shear_prefix}s2n', snr_edges, delta_gamma, cut_source_bin=True, shear_catalog_type=self.config['shear_catalog_type'])

        while True:
            # This happens when we have loaded a new data chunk
            data = yield

            # Indicates the end of the data stream
            if data is None:
                break

            binnedShear.add_data(data)

        
        mu, mean1, mean2, std1, std2 = binnedShear.collect(self.comm)
        print(mu, mean1, mean2, std1, std2)
        if self.rank != 0:
            return

        # Get the error on the mean
        dx = 0.05*(snr_edges[1] - snr_edges[0])
        good = (mu>0) & (np.isfinite(mean1))
        if np.count_nonzero(good)>0:
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(np.log10(mu[good]),mean1[good])
            line1 = slope1*(np.log10(mu))+intercept1
        else:
            line1 = np.zeros(len(mu))
            slope1 = 0
            std_err1 = 0
        good = (mu>0) & (np.isfinite(mean2))
        if np.count_nonzero(good)>0:
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(np.log10(mu[good]),mean2[good])
            line2 = slope2*(np.log10(mu))+intercept2
        else:
            line2 = np.zeros(len(mu))
            slope2 = 0
            std_err2 = 0

        fig = self.open_output('g_snr', wrapper=True)
        
        plt.plot(mu,line1,color='red',label=r"$m=%.4f \pm %.4f$" %(slope1, std_err1))
        plt.plot(mu,[0]*len(mu),color='black')
        plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+',color='red')

        plt.plot(mu,line2,color='blue',label=r"$m=%.4f \pm %.4f$" %(slope2, std_err2))
        plt.plot(mu,[0]*len(mu-dx),color='black')
        plt.xscale('log')
        # plt.ylim(-0.0015,0.0015)
        plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+',color='blue')
        plt.xlabel("SNR")
        plt.ylabel("Mean g")
        plt.legend()
        plt.tight_layout()
        fig.close()

    def plot_size_shear(self):
        # mean shear in bins of galaxy size
        print("Making mean shear galaxy size plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        
        delta_gamma = self.config['delta_gamma']
        psf_prefix = self.config['psf_prefix']
        
        size = 5
        
        T_edges = np.linspace(0.1,2.1,size+1)
        shear_prefix = self.config['shear_prefix']
        binnedShear = MeanShearInBins(f'{shear_prefix}T', T_edges, delta_gamma, cut_source_bin=True, shear_catalog_type=self.config['shear_catalog_type'])

        while True:
            # This happens when we have loaded a new data chunk
            data = yield

            # Indicates the end of the data stream
            if data is None:
                break

            binnedShear.add_data(data)

        mu, mean1, mean2, std1, std2 = binnedShear.collect(self.comm)

        
        if self.rank != 0:
            return

        dx = 0.05*(T_edges[1] - T_edges[0])
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(np.log10(mu),mean1)
        line1 = slope1*(np.log10(mu))+intercept1
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(np.log10(mu),mean2)
        line2 = slope2*(np.log10(mu))+intercept2
        fig = self.open_output('g_T', wrapper=True)
        
        plt.plot(mu,line1,color='red',label=r"$m=%.4f \pm %.4f$" %(slope1, std_err1))
        plt.plot(mu,[0]*len(mu),color='black')
        plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+',color='red')
        
        plt.plot(mu,line2,color='blue',label=r"$m=%.4f \pm %.4f$" %(slope2, std_err2))
        plt.plot(mu,[0]*len(mu),color='black')
        plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+',color='blue')
        #plt.ylim(-0.0015,0.0015)
        plt.xscale('log')
        plt.xlabel("galaxy size T")
        plt.ylabel("Mean g")
        plt.legend()
        plt.tight_layout()
        fig.close()


    def plot_g_histogram(self):
        print('plotting histogram')
        import matplotlib.pyplot as plt
        from scipy import stats
        
        delta_gamma = self.config['delta_gamma']
        bins = 10
        edges = np.linspace(-1, 1, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelMeanVariance(bins)
        calc2 = ParallelMeanVariance(bins)
        
        
        while True:
            data = yield

            if data is None:
                break
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1
        
            if self.config['shear_catalog_type']=='metacal':
                b1 = np.digitize(data['mcal_g1'][qual_cut], edges) - 1
                b1_1p = np.digitize(data['mcal_g1_1p'][qual_cut], edges) - 1 
                b1_2p = np.digitize(data['mcal_g1_2p'][qual_cut], edges) - 1
                b1_1m = np.digitize(data['mcal_g1_1m'][qual_cut], edges) - 1 
                b1_2m = np.digitize(data['mcal_g1_2m'][qual_cut], edges) - 1
            else:
                b1 = np.digitize(data['g1'][qual_cut], edges) - 1 
            
            if self.config['shear_catalog_type']=='metacal':
                b2 = np.digitize(data['mcal_g2'][qual_cut], edges) - 1
                b2_1p = np.digitize(data['mcal_g2_1p'][qual_cut], edges) - 1 
                b2_2p = np.digitize(data['mcal_g2_2p'][qual_cut], edges) - 1
                b2_1m = np.digitize(data['mcal_g2_1m'][qual_cut], edges) - 1 
                b2_2m = np.digitize(data['mcal_g2_2m'][qual_cut], edges) - 1
            else:
                b2 = np.digitize(data['g2'][qual_cut], edges) - 1

            for i in range(bins):
                w1 = np.where(b1==i)
                
                if self.config['shear_catalog_type']=='metacal':
                    w1_1p = np.where(b1_1p==i)
                    w1_2p = np.where(b1_2p==i)
                    w1_1m = np.where(b1_1m==i)
                    w1_2m = np.where(b1_2m==i)
                    S = calculate_selection_response(data['mcal_g1'][qual_cut], data['mcal_g2'][qual_cut], w1_1p, w1_2p,w1_1m, w1_2m, delta_gamma)
                    R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                                  data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                    g1, g2 = apply_metacal_response(R, S, data['mcal_g1'][qual_cut][w1], data['mcal_g2'][qual_cut][w1])
                elif self.config['shear_catalog_type']=='lensfit':
                    g1, g2, weight, one_plus_K = apply_lensfit_calibration(data['g1'][qual_cut][w1], data['g2'][qual_cut][w1],data['weight'][qual_cut][w1])
                elif self.config['shear_catalog_type']=='hsc':
                    g1, g2, weight, one_plus_K = apply_hsc_calibration(data['g1'][qual_cut][w1], data['g2'][qual_cut][w1], data['weight'][qual_cut][w1],
                                                                      c1=data['c1'][qual_cut][w1], c2=data['c2'][qual_cut][w1], sigma_e=data['sigma_e'][qual_cut][w1],
                                                                      m=data['m'][qual_cut][w1])
                else:
                    raise ValueError(f"Please specify metacal or lensfit for shear_catalog in config.")
                # Do more things here to establish
                calc1.add_data(i, g1)
                
                
                w2 = np.where(b2==i)
                
                if self.config['shear_catalog_type']=='metacal':
                    w2_1p = np.where(b2_1p==i)
                    w2_2p = np.where(b2_2p==i)
                    w2_1m = np.where(b2_1m==i)
                    w2_2m = np.where(b2_2m==i)
                    S = calculate_selection_response(data['mcal_g1'][qual_cut], data['mcal_g2'][qual_cut], w1_1p, w1_2p,w1_1m, w1_2m, delta_gamma)
                    R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                                  data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                    g1, g2 = apply_metacal_response(R, S, data['mcal_g1'][qual_cut][w1], data['mcal_g2'][qual_cut][w1])
                elif self.config['shear_catalog_type']=='lensfit':
                    g1, g2, weight, one_plus_K = apply_lensfit_calibration(data['g1'][qual_cut][w1], data['g2'][qual_cut][w1],data['weight'][qual_cut][w1])
                elif self.config['shear_catalog_type']=='hsc':
                    g1, g2, weight, one_plus_K = apply_hsc_calibration(data['g1'][qual_cut][w1], data['g2'][qual_cut][w1], data['weight'][qual_cut][w1],
                                                                      c1=data['c1'][qual_cut][w1], c2=data['c2'][qual_cut][w1], sigma_e=data['sigma_e'][qual_cut][w1],
                                                                      m=data['m'][qual_cut][w1])
                else:
                    raise ValueError(f"Please specify metacal or lensfit for shear_catalog in config.")
                calc2.add_data(i, g2)

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        if self.rank != 0:
            return
        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)
        fig = self.open_output('g1_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1]-edges[0],edgecolor='black',align='center',color='blue')
        plt.xlabel("g1")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()

        fig = self.open_output('g2_hist', wrapper=True)
        plt.bar(mids, count2, width=edges[1]-edges[0], align='center',edgecolor='black',color='purple')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("g2")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count2))
        fig.close()

    def plot_snr_histogram(self):
        print('plotting snr histogram')
        import matplotlib.pyplot as plt
        
        delta_gamma = self.config['delta_gamma']
        shear_prefix = self.config['shear_prefix']
        bins = 10
        edges = np.logspace(1, 3, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelMeanVariance(bins)
        
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1

            b1 = np.digitize(data[f'{shear_prefix}s2n'][qual_cut], edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data[f'{shear_prefix}s2n'][qual_cut][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        if self.rank != 0:
            return
        std1 = np.sqrt(var1/count1)
        fig = self.open_output('snr_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1:]-edges[:-1],edgecolor='black',align='center',color='blue')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xscale('log')
        plt.xlabel("log(snr)")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()

    def plot_response_histograms(self):
        import matplotlib.pyplot as plt
        if self.comm:
            import mpi4py.MPI
        size = 10
        # This seems to be a reasonable range, though there are samples
        # with extremely high values
        edges = np.linspace(-3, 3, size+1)
        mid = 0.5*(edges[1:] + edges[:-1])
        width = edges[1] - edges[0]
        if self.config['shear_catalog_type']=='metacal':
            # count of objects
            counts = np.zeros((2,2,size))
            # make a separate histogram of the shear-sample-selected
            # objects
            counts_s = np.zeros((2,2,size))
        else:
            # count of objects
            counts = np.zeros((size))
            # make a separate histogram of the shear-sample-selected
            # objects
            counts_s = np.zeros((size))
        while True:
            data = yield

            if data is None:
                break

            # check if selected for any source bin
            in_shear_sample = data['source_bin'] !=-1
            if self.config['shear_catalog_type']=='metacal':
                B = np.digitize(data['R_gamma'], edges) - 1
                # loop through this chunk of data.
                for s, b in zip(in_shear_sample, B):
                    # for each element in the 2x2 matrix
                    for i in range(2):
                        for j in range(2):
                            bij = b[i, j]
                            # this will naturally filter out
                            # the nans
                            if (bij >= 0) and (bij < size):
                                counts[i, j, bij] += 1
                                if s:
                                    counts_s[i, j, bij] += 1
            else:
                B = np.digitize(data['R'], edges) - 1
                # loop through this chunk of data.
                for s, b in zip(in_shear_sample, B):
                    if (b >= 0) and (b < size):
                        counts[b] +=1
                        if s:
                            counts_s[b] += 1

        # Sum from all processors and then non-root ones return
        if self.comm is not None:
            if self.rank == 0:
                self.comm.Reduce(mpi4py.MPI.IN_PLACE, counts)
                self.comm.Reduce(mpi4py.MPI.IN_PLACE, counts_s)
            else:
                self.comm.Reduce(counts, None)
                self.comm.Reduce(counts_s, None)

                # only root process makes plots
                return


        fig = self.open_output('response_hist', wrapper=True, figsize=(10, 5))

        plt.subplot(1,2,1)
        if self.config['shear_catalog_type']=='metacal':
            manual_step_histogram(edges, counts[0, 0], label='R00', color='#1f77b4')
            manual_step_histogram(edges, counts[1, 1], label='R11', color='#ff7f0e')
            manual_step_histogram(edges, counts[0, 1], label='R01', color='#2ca02c')
            manual_step_histogram(edges, counts[1, 0], label='R10', color='#d62728')
        else:
            manual_step_histogram(edges, counts, label='R', color='#1f77b4')
        plt.ylim(0, counts.max()*1.1)
        plt.xlabel("R_gamma")
        plt.ylabel("Count")
        plt.title("All flag=0")

        plt.subplot(1,2,2)
        if self.config['shear_catalog_type']=='metacal':
            manual_step_histogram(edges, counts_s[0, 0], label='R00', color='#1f77b4')
            manual_step_histogram(edges, counts_s[1, 1], label='R11', color='#ff7f0e')
            manual_step_histogram(edges, counts_s[0, 1], label='R01', color='#2ca02c')
            manual_step_histogram(edges, counts_s[1, 0], label='R10', color='#d62728')
        else:
            manual_step_histogram(edges, counts_s, label='R', color='#1f77b4')
        plt.ylim(0, counts_s.max()*1.1)
        plt.xlabel("R_gamma")
        plt.ylabel("Count")
        plt.title("Source sample")

        plt.legend()
        plt.tight_layout()
        fig.close()



    def plot_mag_histograms(self):
        if self.comm:
            import mpi4py.MPI
        # mean shear in bins of PSF
        print("Making mag histogram")
        import matplotlib.pyplot as plt
        size = 10
        mag_min = 20
        mag_max = 30
        edges = np.linspace(mag_min, mag_max, size+1)
        mid = 0.5*(edges[1:] + edges[:-1])
        width = edges[1] - edges[0]
        bands = 'ugrizy'
        nband = len(bands)
        full_hists = [np.zeros(size, dtype=int) for b in bands]
        source_hists = [np.zeros(size, dtype=int) for b in bands]


        while True:
            data = yield

            if data is None:
                break

            for (b, h1,h2) in zip(bands, full_hists, source_hists):
                b1 = np.digitize(data[f'mag_{b}'], edges) - 1


                for i in range(size):
                    w = b1==i
                    count = w.sum()
                    h1[i] += count

                    w &= (data['source_bin']>=0)
                    count = w.sum()
                    h2[i] += count

        if self.comm is not None:
            full_hists = reduce(self.comm, full_hists)
            source_hists = reduce(self.comm, source_hists)

        if self.rank == 0:
            fig = self.open_output('mag_hist', wrapper=True, figsize=(4,nband*3))
            for i, (b,h1,h2) in enumerate(zip(bands, full_hists, source_hists)):
                plt.subplot(nband, 1, i+1)
                plt.bar(mid, h1, width=width, fill=False,  label='Complete', edgecolor='r')
                plt.bar(mid, h2, width=width, fill=True,  label='WL Source', color='g')
                plt.xlabel(f"Mag {b}")
                plt.ylabel("N")
                if i==0:
                    plt.legend()
            plt.tight_layout()
            fig.close()


def reduce(comm, H):
    H2 = []
    rank = comm.Get_rank()
    for	h in H:
        if rank == 0:
            hsum = np.zeros_like(h)
        else:
            hsum = None
        comm.Reduce(h, hsum)
        H2.append(hsum)
    return H2


