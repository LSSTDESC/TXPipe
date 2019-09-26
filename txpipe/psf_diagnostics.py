from .base_stage import PipelineStage
from .data_types import Directory, HDFFile, PNGFile, TomographyCatalog
from .utils.stats import ParallelStatsCalculator, combine_variances
import numpy as np

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
        ('T_psf_residual_hist', PNGFile)

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
        chunk_rows = 10000
        star_cols = ['measured_e1','model_e1','measured_e2','model_e2','measured_T','model_T']
        iter_star = self.iterate_hdf('star_catalog','stars',star_cols,chunk_rows)

        # Now loop through each chunk of input data, one at a time.
        # Each time we get a new segment of data, which goes to all the plotters
        for (start, end, data), in zip(iter_star):
            print(f"Read data {start} - {end}")
            # This causes each data = yield statement in each plotter to
            # be given this data chunk as the variable data.
            #data.update(data2)
            #data.update(data3)
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
                      
    def plot_psf_e1_residual_histogram(self):
        # general plotter for histograms
        # TODO think about a smart way to define the bin numbers, also
        # make this more general for all quantities
        print('plotting psf e1 residual histogram')
        import matplotlib.pyplot as plt
        bins = 50
        edges = np.linspace(-10, 10, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        while True:
            data = yield

            if data is None:
                break

            b1 = np.digitize((data['measured_e1']-data['model_e1'])/data['measured_e1'], edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, (data['measured_e1'][w]-data['model_e1'][w])/data['measured_e1'][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        std1 = np.sqrt(var1/count1)
        if self.rank != 0:
            return
        fig = self.open_output('e1_psf_residual_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1]-edges[0],edgecolor='black',align='center',color='blue')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("($e_{1}-e_{1,psf})/e_{1,psf}$")
        plt.ylabel(r'$N_{stars}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()

    def plot_psf_e2_residual_histogram(self):
        # general plotter for histograms
        # TODO think about a smart way to define the bin numbers, also
        # make this more general for all quantities
        print('plotting psf e2 residual histogram')
        import matplotlib.pyplot as plt
        bins = 50
        edges = np.linspace(-10, 10, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        while True:
            data = yield

            if data is None:
                break

            b1 = np.digitize((data['measured_e2']-data['model_e2'])/data['measured_e2'], edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, (data['measured_e2'][w]-data['model_e2'][w])/data['measured_e2'][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        std1 = np.sqrt(var1/count1)
        if self.rank != 0:
            return
        fig = self.open_output('e2_psf_residual_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1]-edges[0],edgecolor='black',align='center',color='blue')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("$(e_{2}-e_{2,psf})/e_{2,psf}$")
        plt.ylabel(r'$N_{stars}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()

    def plot_psf_T_residual_histogram(self):
        # general plotter for histograms
        # TODO think about a smart way to define the bin numbers, also
        # make this more general for all quantities
        print('plotting psf T residual histogram')
        import matplotlib.pyplot as plt
        bins = 50
        edges = np.linspace(-.1, .1, bins+1)
        mids = 0.5*(edges[1:] + edges[:-1])
        calc1 = ParallelStatsCalculator(bins)
        while True:
            data = yield

            if data is None:
                break

            b1 = np.digitize((data['measured_T']-data['model_T'])/data['measured_T'], edges) - 1

            for i in range(bins):
                w = np.where(b1==i)
                # Do more things here to establish
                calc1.add_data(i, (data['measured_T'][w]-data['model_T'][w])/data['measured_T'][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode='gather')
        std1 = np.sqrt(var1/count1)
        if self.rank != 0:
            return
        fig = self.open_output('T_psf_residual_hist', wrapper=True)
        plt.bar(mids, count1, width=edges[1]-edges[0],edgecolor='black',align='center',color='blue')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("$(T-T_{psf})/T_{psf}$")
        plt.ylabel(r'$N_{stars}$')
        plt.ylim(0,1.1*max(count1))
        fig.close()

