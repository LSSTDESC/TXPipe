from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile, TomographyCatalog
from .utils.stats import ParallelStatsCalculator
from .utils.metacal import calculate_selection_response, calculate_shear_response, apply_metacal_response
import numpy as np

class TXDiagnosticPlots(PipelineStage):
    """
    """
    name='TXDiagnosticPlots'

    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_catalog', HDFFile),
        ('tomography_catalog', TomographyCatalog),
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

    ]

    config_options = {
        'chunk_rows': 100000,
        'delta_gamma': 0.02
    }

    def run(self):
        # PSF tests
        import matplotlib
        matplotlib.use('agg')

        # Collect together all the methods on this class called self.plot_*
        # They are all expected to be python coroutines - generators that
        # use the yield feature to pause and wait for more input.
        # We instantiate them all here
        
        plotters = [getattr(self, f)() for f in dir(self) if f.startswith('plot_')]

        # Start off each of the plotters.  This will make them all run up to the
        # first yield statement, then pause and wait for the first chunk of data
        for plotter in plotters:
            print(plotter)
            plotter.send(None)

        # Create an iterator for reading through the input data.
        # This method automatically splits up data among the processes,
        # so the plotters should handle this.
        chunk_rows = self.config['chunk_rows']
        shear_cols = ['mcal_psf_g1', 'mcal_psf_g2','mcal_g1','mcal_g1_1p','mcal_g1_2p','mcal_g1_1m','mcal_g1_2m','mcal_g2','mcal_g2_1p','mcal_g2_2p','mcal_g2_1m','mcal_g2_2m','mcal_psf_T_mean','mcal_s2n','mcal_T',
                     'mcal_T_1p','mcal_T_2p','mcal_T_1m','mcal_T_2m','mcal_s2n_1p','mcal_s2n_2p','mcal_s2n_1m',
                     'mcal_s2n_2m']
        iter_shear = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)

        photo_cols = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']
        iter_phot = self.iterate_hdf('photometry_catalog', 'photometry', photo_cols, chunk_rows)

        tomo_cols = ['source_bin','lens_bin']
        iter_tomo = self.iterate_hdf('tomography_catalog','tomography',tomo_cols,chunk_rows)

        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for (start, end, data), (_, _, data2), (_, _, data3) in zip(iter_shear, iter_tomo, iter_phot):
            print(f"Read data {start} - {end}")
            # This causes each data = yield statement in each plotter to
            # be given this data chunk as the variable data.
            data.update(data2)
            data.update(data3)
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
        
        delta_gamma = self.config['delta_gamma']
        size = 11
        psf_g_edges = np.linspace(-0.024, 0.044, size+1)
        psf_g_mid = 0.5*(psf_g_edges[1:] + psf_g_edges[:-1])
        calc11 = ParallelStatsCalculator(size)
        calc12 = ParallelStatsCalculator(size)
        calc21 = ParallelStatsCalculator(size)
        calc22 = ParallelStatsCalculator(size)
        mu1 = ParallelStatsCalculator(size)
        mu2 = ParallelStatsCalculator(size)
        while True:
            data = yield

            if data is None:
                break
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1
    
            b1 = np.digitize(data['mcal_psf_g1'][qual_cut], psf_g_edges) - 1
                
            b2 = np.digitize(data['mcal_psf_g2'][qual_cut], psf_g_edges) - 1

            for i in range(size):
                w1 = np.where(b1==i)   
                
                w2 = np.where(b2==i)
                
                # Calculate the shear response (no selection bias for psf quantities)
                R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                        data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                
                g1_1, g2_1 = apply_metacal_response(R, 0, data['mcal_g1'][qual_cut][w1], data['mcal_g2'][qual_cut][w1]) 
                
                calc11.add_data(i, g1_1)
                calc12.add_data(i, g2_1)
                
                # Calculate the shear response 
                R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                        data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                
                g1_2, g2_2 = apply_metacal_response(R, 0, data['mcal_g1'][qual_cut][w2], data['mcal_g2'][qual_cut][w2])
                
                calc21.add_data(i, g1_2)
                calc22.add_data(i, g2_2)
                mu1.add_data(i, data['mcal_psf_g1'][qual_cut][w1])
                mu2.add_data(i, data['mcal_psf_g2'][qual_cut][w2])
        count11, mean11, var11 = calc11.collect(self.comm, mode='gather')
        count12, mean12, var12 = calc12.collect(self.comm, mode='gather')
        count21, mean21, var21 = calc21.collect(self.comm, mode='gather')
        count22, mean22, var22 = calc22.collect(self.comm, mode='gather')

        _, mu1, _ = mu1.collect(self.comm, mode='gather')
        _, mu2, _ = mu2.collect(self.comm, mode='gather')

        if self.rank != 0:
            return

        std11 = np.sqrt(var11/count11)
        std12 = np.sqrt(var12/count12)
        std21 = np.sqrt(var21/count21)
        std22 = np.sqrt(var22/count22)

        fig = self.open_output('g_psf_g', wrapper=True)
        #Include a small shift to be able to see the g1 / g2 points on the plot
        dx = 0.1*(mu1[1] - mu1[0])

        slope11, intercept11, r_value11, p_value11, std_err11 = stats.linregress(mu1,mean11)
        line11 = slope11*(mu1)+intercept11

        slope12, intercept12, r_value12, p_value12, std_err12 = stats.linregress(mu1,mean12)
        line12 = slope12*(mu1)+intercept12

        slope21, intercept21, r_value21, p_value21, std_err21 = stats.linregress(mu2,mean21)
        line21 = slope21*(mu2)+intercept21

        slope22, intercept22, r_value22, p_value22, std_err22 = stats.linregress(mu2,mean22)
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

        # This also saves the figure
        fig.close()

    def plot_psf_size_shear(self):
        # mean shear in bins of PSF
        print("Making PSF size plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        
        delta_gamma = self.config['delta_gamma']
        size = 11
        psf_g_edges = np.linspace(0.3, 0.6, size+1)
        psf_g_mid = 0.5*(psf_g_edges[1:] + psf_g_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
            
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1
    
            b1 = np.digitize(data['mcal_psf_T_mean'][qual_cut], psf_g_edges) - 1

            for i in range(size):
                w = np.where(b1==i)
                
                # Calculate the shear response 
                R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                                  data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                
                g1, g2 = apply_metacal_response(R, 0, data['mcal_g1'][qual_cut][w], data['mcal_g2'][qual_cut][w])
                
                # Do more things here to establish
                calc1.add_data(i, g1)
                calc2.add_data(i, g2)
                mu.add_data(i, data['mcal_psf_T_mean'][qual_cut][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')

        if self.rank == 0:
            std1 = np.sqrt(var1/count1)
            std2 = np.sqrt(var2/count2)
            
            dx = 0.05*(psf_g_mid[1] - psf_g_mid[0])
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(mu,mean1)
            line1 = slope1*(mu)+intercept1
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(mu,mean2)
            line2 = slope2*(mu)+intercept2

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
            plt.ylim(-0.0015,0.0015)
            plt.legend(loc='best')
            plt.tight_layout()
            fig.close()

    def plot_snr_shear(self):
        # mean shear in bins of snr
        print("Making mean shear SNR plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        
        delta_gamma = self.config['delta_gamma']
        size = 10
        snr_edges = np.logspace(.1,2.5,size+1)
        snr_mid = 0.5*(snr_edges[1:] + snr_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
        
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1

            b1 = np.digitize(data['mcal_s2n'][qual_cut], snr_edges) - 1
            b_1p = np.digitize(data['mcal_s2n_1p'][qual_cut], snr_edges) - 1 
            b_2p = np.digitize(data['mcal_s2n_2p'][qual_cut], snr_edges) - 1
            b_1m = np.digitize(data['mcal_s2n_1m'][qual_cut], snr_edges) - 1 
            b_2m = np.digitize(data['mcal_s2n_2m'][qual_cut], snr_edges) - 1

            for i in range(size):
                w = np.where(b1==i)
                w_1p = np.where(b_1p==i)
                w_2p = np.where(b_2p==i)
                w_1m = np.where(b_1m==i)
                w_2m = np.where(b_2m==i)
                
                
                # Calculate the shear response 
                S = calculate_selection_response(data['mcal_g1'][qual_cut], data['mcal_g2'][qual_cut], w_1p, w_2p,w_1m, w_2m, delta_gamma)
                R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                                  data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                
                g1, g2 = apply_metacal_response(R, S, data['mcal_g1'][qual_cut][w], data['mcal_g2'][qual_cut][w])
                # Do more things here to establish
                calc1.add_data(i, g1)
                calc2.add_data(i, g2)
                mu.add_data(i, data['mcal_s2n'][qual_cut][w])
                           
        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')

        if self.rank == 0:
            std1 = np.sqrt(var1/count1)
            std2 = np.sqrt(var2/count2)
            
            dx = 0.05*(snr_mid[1] - snr_mid[0])
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(np.log10(mu),mean1)
            line1 = slope1*(np.log10(mu))+intercept1
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(np.log10(mu),mean2)
            line2 = slope2*(np.log10(mu))+intercept2
            fig = self.open_output('g_snr', wrapper=True)
            
            plt.plot(mu,line1,color='red',label=r"$m=%.4f \pm %.4f$" %(slope1, std_err1))
            plt.plot(mu,[0]*len(mu),color='black')
            plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+',color='red')

            plt.plot(mu,line2,color='blue',label=r"$m=%.4f \pm %.4f$" %(slope2, std_err2))
            plt.plot(mu,[0]*len(mu-dx),color='black')
            plt.xscale('log')
            plt.ylim(-0.0015,0.0015)
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
        
        size = 10
        T_edges = np.linspace(0.1,2.1,size+1)
        T_mid = 0.5*(T_edges[1:] + T_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
        
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1

            b = np.digitize(data['mcal_T'][qual_cut], T_edges) - 1
            b_1p = np.digitize(data['mcal_T_1p'][qual_cut], T_edges) - 1 
            b_2p = np.digitize(data['mcal_T_2p'][qual_cut], T_edges) - 1
            b_1m = np.digitize(data['mcal_T_1m'][qual_cut], T_edges) - 1 
            b_2m = np.digitize(data['mcal_T_2m'][qual_cut], T_edges) - 1

            for i in range(size):
                w = np.where(b==i)
                w_1p = np.where(b_1p==i)
                w_2p = np.where(b_2p==i)
                w_1m = np.where(b_1m==i)
                w_2m = np.where(b_2m==i)
                # Calculate the shear response 
                S = calculate_selection_response(data['mcal_g1'][qual_cut], data['mcal_g2'][qual_cut], w_1p, w_2p,w_1m, w_2m, delta_gamma)
                R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                                  data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                
                g1, g2 = apply_metacal_response(R, S, data['mcal_g1'][qual_cut][w], data['mcal_g2'][qual_cut][w])
                
                calc1.add_data(i, g1)
                calc2.add_data(i, g2)
                mu.add_data(i, data['mcal_T'][qual_cut][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')
        
        if self.rank == 0:
            std1 = np.sqrt(var1/count1)
            std2 = np.sqrt(var2/count2)
            
            dx = 0.05*(T_mid[1] - T_mid[0])
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
            plt.ylim(-0.0015,0.0015)
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
        bins = 50
        edges = np.linspace(-1, 1, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        calc2 = ParallelStatsCalculator(bins)
        
        
        while True:
            data = yield

            if data is None:
                break
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1
        
            b1 = np.digitize(data['mcal_g1'][qual_cut], edges) - 1
            b1_1p = np.digitize(data['mcal_g1_1p'][qual_cut], edges) - 1 
            b1_2p = np.digitize(data['mcal_g1_2p'][qual_cut], edges) - 1
            b1_1m = np.digitize(data['mcal_g1_1m'][qual_cut], edges) - 1 
            b1_2m = np.digitize(data['mcal_g1_2m'][qual_cut], edges) - 1
            
            b2 = np.digitize(data['mcal_g2'][qual_cut], edges) - 1
            b2_1p = np.digitize(data['mcal_g2_1p'][qual_cut], edges) - 1 
            b2_2p = np.digitize(data['mcal_g2_2p'][qual_cut], edges) - 1
            b2_1m = np.digitize(data['mcal_g2_1m'][qual_cut], edges) - 1 
            b2_2m = np.digitize(data['mcal_g2_2m'][qual_cut], edges) - 1

            for i in range(bins):
                w1 = np.where(b1==i)
                w1_1p = np.where(b1_1p==i)
                w1_2p = np.where(b1_2p==i)
                w1_1m = np.where(b1_1m==i)
                w1_2m = np.where(b1_2m==i)
                
                S = calculate_selection_response(data['mcal_g1'][qual_cut], data['mcal_g2'][qual_cut], w1_1p, w1_2p,w1_1m, w1_2m, delta_gamma)
                R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                                  data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                
                g1, g2 = apply_metacal_response(R, S, data['mcal_g1'][qual_cut][w1], data['mcal_g2'][qual_cut][w1])
                # Do more things here to establish
                calc1.add_data(i, g1)
                
                
                w2 = np.where(b2==i)
                w2_1p = np.where(b2_1p==i)
                w2_2p = np.where(b2_2p==i)
                w2_1m = np.where(b2_1m==i)
                w2_2m = np.where(b2_2m==i)
                
                S = calculate_selection_response(data['mcal_g1'][qual_cut], data['mcal_g2'][qual_cut], w2_1p, w2_2p,w2_1m, w2_2m, delta_gamma)
                R = calculate_shear_response(data['mcal_g1_1p'][qual_cut],data['mcal_g1_2p'][qual_cut],data['mcal_g1_1m'][qual_cut],data['mcal_g1_2m'][qual_cut],
                                                  data['mcal_g2_1p'][qual_cut],data['mcal_g2_2p'][qual_cut],data['mcal_g2_1m'][qual_cut],data['mcal_g2_2m'][qual_cut],delta_gamma)
                
                g1, g2 = apply_metacal_response(R, S, data['mcal_g1'][qual_cut][w2], data['mcal_g2'][qual_cut][w2])
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
        bins = 50
        edges = np.logspace(1, 3, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        
        while True:
            data = yield

            if data is None:
                break
            
            qual_cut = data['source_bin'] !=-1
#            qual_cut |= data['lens_bin'] !=-1

            b1 = np.digitize(data['mcal_s2n'][qual_cut], edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_s2n'][qual_cut][w])

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

    def plot_mag_histograms(self):
        if self.comm:
            import mpi4py.MPI
        # mean shear in bins of PSF
        print("Making mag histogram")
        import matplotlib.pyplot as plt
        size = 20
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
                b1 = np.digitize(data[f'{b}_mag'], edges) - 1


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
