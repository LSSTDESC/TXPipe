from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile
from .utils.stats import ParallelStatsCalculator
import numpy as np

class TXInputDiagnostics(PipelineStage):
    """
    """
    name='TXInputDiagnostics'

    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_catalog', HDFFile),
    ]
    outputs = [
        ('g_psf_T', PNGFile),
        ('g_psf_g', PNGFile),
        ('mag_hist', PNGFile),
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
            plotter.send(None)

        # Create an iterator for reading through the input data.
        # This method automatically splits up data among the processes,
        # so the plotters should handle this.
        chunk_rows = 10000
        shear_cols = ['mcal_psf_g1', 'mcal_psf_g2', 'mcal_g1', 'mcal_g2', 'mcal_psf_T_mean']
        photo_cols = ['mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_y']
        iter_shear = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        iter_phot = self.iterate_hdf('photometry_catalog', 'photometry', photo_cols, chunk_rows)

        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for (start, end, data), (_, _, data2) in iter_shear:
            print(f"Read data {start} - {end}")
            data.update(data2)
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
        hists = [np.zeros(size, dtype=int) for b in bands]


        while True:
            data = yield

            if data is None:
                break

            for b, h in zip(bands, hists):
                b1 = np.digitize(data[f'mag_{b}'], edges) - 1

                for i in range(size):
                    w = np.where(b1==i)
                    count = w[0].size
                    h[i] += count

        for h in hists:
            if self.comm:
                self.comm.Reduce(None, h)

        if self.rank == 0:
            fig = self.open_output('mag_hist', wrapper=True, figsize=(4,nbands*4))
            for i, (b,h) in enumerate(zip(bands, hists)):
                plt.subplot(nbands, 1, i+1)
                plt.bar(mid, h, width=width)
                plt.xlabel(f"Mag {b}")
                plt.ylabel("N")
            plt.tight_layout()
            fig.close()
