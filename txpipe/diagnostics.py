from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile, TomographyCatalog
from .utils.stats import ParallelStatsCalculator, combine_variances
import numpy as np

class TXDiagnostics(PipelineStage):
    """
    """
    name='TXDiagnostics'

    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_catalog', HDFFile),
        ('tomography_catalog', TomographyCatalog)
    ]
    outputs = [
        ('g_psf_T', PNGFile),
        ('g_psf_g', PNGFile),
        ('g1_hist', PNGFile),
        ('g2_hist', PNGFile),
        ('g_snr', PNGFile)
    ]
    config = {}

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
        #TODO exactly what SNR do we want here
        chunk_rows = 10000
        shear_cols = ['mcal_psf_g1', 'mcal_psf_g2', 'mcal_g1', 'mcal_g2', 'mcal_psf_T_mean','mcal_s2n']
        iter_shear = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        tomo_cols = ['R_S','R_gamma']
        iter_tomo = self.iterate_hdf('tomography_catalog','multiplicative_bias',tomo_cols,chunk_rows)

        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for start, end, data in iter_shear:
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
        size = 11
        psf_g_edges = np.linspace(-5e-5, 5e-5, size+1)
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

            b1 = np.digitize(data['mcal_psf_g1'], psf_g_edges) - 1
            b2 = np.digitize(data['mcal_psf_g2'], psf_g_edges) - 1

            for i in range(size):
                w1 = np.where(b1==i)
                w2 = np.where(b2==i)

                # Do more things here to establish
                calc11.add_data(i, data['mcal_g1'][w1])
                calc12.add_data(i, data['mcal_g2'][w1])
                calc21.add_data(i, data['mcal_g1'][w2])
                calc22.add_data(i, data['mcal_g2'][w2])
                mu1.add_data(i, data['mcal_psf_g1'][w1])
                mu2.add_data(i, data['mcal_psf_g2'][w2])
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
        dx = 0.1*(mu1[1] - mu1[0])

        plt.subplot(2,1,1)
        plt.errorbar(mu1+dx, mean11, std11, label='g1', fmt='+')
        plt.errorbar(mu1-dx, mean12, std12, label='g2', fmt='+')
        plt.xlabel("PSF g1")
        plt.ylabel("Mean g")
        plt.legend()

        plt.subplot(2,1,2)
        plt.errorbar(mu2+dx, mean21, std21, label='g1', fmt='+')
        plt.errorbar(mu2-dx, mean22, std22, label='g2', fmt='+')
        plt.legend()
        plt.xlabel("PSF g2")
        plt.ylabel("Mean g")
        plt.tight_layout()

        # This also saves the figure
        fig.close()

    def plot_psf_size_shear(self):
        # mean shear in bins of PSF
        print("Making PSF size plot")
        import matplotlib.pyplot as plt
        size = 11
        psf_g_edges = np.linspace(0.18, 0.19, size+1)
        psf_g_mid = 0.5*(psf_g_edges[1:] + psf_g_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
        while True:
            data = yield

            if data is None:
                break

            b1 = np.digitize(data['mcal_psf_T_mean'], psf_g_edges) - 1

            for i in range(size):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_g1'][w])
                calc2.add_data(i, data['mcal_g2'][w])
                mu.add_data(i, data['mcal_psf_T_mean'][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')

        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)

        dx = 0.05*(psf_g_mid[1] - psf_g_mid[0])
        if self.rank == 0:
            fig = self.open_output('g_psf_T', wrapper=True)
            plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+')
            plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+')
            plt.xlabel("PSF T")
            plt.ylabel("Mean g")
            plt.legend()
            plt.tight_layout()
            fig.close()

    def plot_snr_shear(self):
        # mean shear in bins of PSF
        print("Making mean shear SNR plot")
        import matplotlib.pyplot as plt
        size = 10
        snr_edges = np.logspace(1, 3, size+1)
        snr_mid = 0.5*(snr_edges[1:] + snr_edges[:-1])
        calc1 = ParallelStatsCalculator(size)
        calc2 = ParallelStatsCalculator(size)
        mu = ParallelStatsCalculator(size)
        while True:
            data = yield

            if data is None:
                break

            b1 = np.digitize(data['mcal_s2n'], snr_edges) - 1

            for i in range(size):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_g1'][w])
                calc2.add_data(i, data['mcal_g2'][w])
                mu.add_data(i, data['mcal_s2n'][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        _, mu, _ = mu.collect(self.comm, mode='gather')

        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)

        dx = 0.05*(snr_mid[1] - snr_mid[0])
        if self.rank == 0:
            fig = self.open_output('g_snr', wrapper=True)
            plt.errorbar(mu+dx, mean1, std1, label='g1', fmt='+')
            plt.errorbar(mu-dx, mean2, std2, label='g2', fmt='+')
            plt.xscale('log')
            plt.xlabel("SNR")
            plt.ylabel("Mean g")
            plt.legend()
            plt.tight_layout()
            fig.close()


    def plot_histogram(self):
        # general plotter for histograms
        # TODO think about a smart way to define the bin numbers, also
        # make this more general for all quantities
        print('plotting histogram')
        import matplotlib.pyplot as plt
        bins = 50
        edges = np.linspace(-1, 1, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        calc2 = ParallelStatsCalculator(bins)
        while True:
            data = yield

            if data is None:
                break

            b1 = np.digitize(data['mcal_g1'], edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, data['mcal_g1'][w])
                calc2.add_data(i, data['mcal_g2'][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        count2, mean2, var2 = calc2.collect(self.comm, mode='gather')
        std1 = np.sqrt(var1/count1)
        std2 = np.sqrt(var2/count2)
        if self.rank != 0:
            return
        fig = self.open_output('g1_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1]-edges[0],edgecolor='black',align='center',color='blue')
        plt.xlabel("g1")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()

        fig = self.open_output('g2_hist', wrapper=True)
        plt.bar(mids, count2, width=edges[1]-edges[0], align='center',edgecolor='black',color='purple')
        plt.xlabel("g2")
        plt.ylabel(r'$N_{galaxies}$')
        plt.ylim(0,1.1*max(count2))
        fig.close()

# gamma t around field centres
# gamma t around

# PSF as function of shear
