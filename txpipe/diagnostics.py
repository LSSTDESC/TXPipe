from .base_stage import PipelineStage
from .data_types import Directory, ShearCatalog, HDFFile, PNGFile, TomographyCatalog, TextFile
from parallel_statistics import ParallelMeanVariance, ParallelHistogram
from .utils.calibrators import Calibrator
from .utils.calibration_tools import (
    calculate_selection_response,
    calculate_shear_response,
    MeanShearInBins,
    read_shear_catalog_type,
    metadetect_variants,
    band_variants,
)
from .utils.fitting import fit_straight_line
from .utils import import_dask
from .plotting import manual_step_histogram
import numpy as np

class TXDiagnosticQuantiles(PipelineStage):
    """
    Measure quantiles of various values in the shear catalog.

    This uses a library called "Distogram" which builds a
    histogram that it gradually updates, tweaking the edges
    as it goes along. This means we don't have to load
    the full data into memory ever.

    The algorithm used can be noisy if there are large outliers in
    the data, but after selection cuts are made this seems to be okay here,
    with all the quantiles for the base quantities (T, s2n, T_psf, g1_psf) within
    10% of the true quantile and the majority within 1%. For our purposes
    (defining bins for diagonstics) this is fine.
    """
    name = "TXDiagnosticQuantiles"
    dask_parallel = True
    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
    ]
    outputs = [
        ("shear_catalog_quantiles", HDFFile),
    ]
    config_options = {
        "shear_prefix": "mcal_",
        "psf_prefix": "mcal_psf_",
        "nbins": 20,
        "chunk_rows": 0,
        "bands": "riz",
    }
    def run(self):
        _, da = import_dask()

        # Configuration parameters
        psf_prefix = self.config["psf_prefix"]
        shear_prefix = self.config["shear_prefix"]
        chunk_rows = self.config["chunk_rows"]
        nedge = self.config["nbins"] + 1
        if chunk_rows == 0:
            chunk_rows = "auto"

        # We canonicalise the names here
        col_names = {
            "psf_g1": f"{psf_prefix}g1",
            "psf_T_mean": f"{psf_prefix}T_mean",
            "s2n": f"{shear_prefix}s2n",
            "T": f"{shear_prefix}T",
        }

        for band in self.config["bands"]:
            col_names[f"mag_{band}"] = f"{shear_prefix}mag_{band}"

        # We ask for quantiles at these points
        quantiles = np.linspace(0, 1, nedge, endpoint=True)
        percentiles = quantiles * 100

        with self.open_input("shear_catalog") as f, self.open_input("shear_tomography_catalog") as g:
            # We will be checking if the source is in a tomographic bin
            # and doing quantiles only of selected obejcts (in any bin)
            bins = da.from_array(g["tomography/bin"], chunks=chunk_rows)
            selected = bins >= 0

            # We now build up the quantile values
            quantile_values = {}
            for new_name, old_name in col_names.items():
                # Create dask arrays of the columns. This loads them lazily,
                # so no data is actually loaded here. Only when the "compute"
                # method is called below does anything actually happen.
                col = da.from_array(f[f"shear/{old_name}"], chunks=chunk_rows)

                # Ask dask to compute the percentiles of this column.
                # Again, it will not actually do anything until the "compute"
                # method is called below. When that happens, it will
                # chunk up the data and calculate the percentiles in parallel.
                quantile_values[new_name] = da.percentile(col[selected], percentiles)

            # Now ask dask to actually do the calculations
            quantile_values, = da.compute(quantile_values)

        # Open the output file and save the results
        with self.open_output("shear_catalog_quantiles") as f:
            # put everything in a group called "quantiles"
            g = f.create_group("quantiles")
            # Save the quantile points
            g.create_dataset("quantiles", data=quantiles)
            # Save the quantile values themselves
            for name, quantile_values in quantile_values.items():
                g.create_dataset(name, data=quantile_values)



class TXSourceDiagnosticPlots(PipelineStage):
    """
    Make diagnostic plots of the shear catalog

    This includes both tomographic and 2D measurements, and includes
    the PSF as a function of various quantities
    """

    name = "TXSourceDiagnosticPlots"

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("shear_catalog_quantiles", HDFFile)
    ]

    outputs = [
        ("g_psf_T", PNGFile),
        ("g_psf_g", PNGFile),
        ("g1_hist", PNGFile),
        ("g2_hist", PNGFile),
        ("g_snr", PNGFile),
        ("g_T", PNGFile),
        ("g_colormag",PNGFile),
        ("source_snr_hist", PNGFile),
        ("source_mag_hist", PNGFile),
        ("response_hist", PNGFile),
        ("g_psf_T_out",TextFile),
        ("g_psf_g_out",TextFile),
        ("g_snr_out",TextFile),
        ("g_T_out",TextFile),
    ]

    config_options = {
        "chunk_rows": 100000,
        "delta_gamma": 0.02,
        "shear_prefix": "mcal_",
        "psf_prefix": "mcal_psf_",
        "nbins": 20,
        "g_min":-0.03,
        "g_max": 0.05,
        "psfT_min": 0.04,
        "psfT_max": 0.36,
        "T_min": 0.04,
        "T_max": 4.0,
        "s2n_min": 10,
        "s2n_max": 300,
        "psf_unit_conv": False,
        "bands": "riz",
    }

    def run(self):
        # PSF tests
        import matplotlib

        matplotlib.use("agg")

        # this also sets self.config["shear_catalog_type"]
        cat_type = read_shear_catalog_type(self)
        
        # Collect together all the methods on this class called self.plot_*
        # They are all expected to be python coroutines - generators that
        # use the yield feature to pause and wait for more input.
        # We instantiate them all here

        plotters = [getattr(self, f)() for f in dir(self) if f.startswith("plot_")]
       
        # Start off each of the plotters.  This will make them all run up to the
        # first yield statement, then pause and wait for the first chunk of data
        for plotter in plotters:
            plotter.send(None)
        # Create an iterator for reading through the input data.
        # This method automatically splits up data among the processes,
        # so the plotters should handle this.
        chunk_rows = self.config["chunk_rows"]
        psf_prefix = self.config["psf_prefix"]
        shear_prefix = self.config["shear_prefix"]
        bands = self.config["bands"]
        if self.rank == 0:
            print("Catalog type = ", cat_type)

        if cat_type == "metacal":
            shear_cols = [
                f"{psf_prefix}g1",
                f"{psf_prefix}g2",
                f"{psf_prefix}T_mean",
                "mcal_g1",
                "mcal_g1_1p",
                "mcal_g1_2p",
                "mcal_g1_1m",
                "mcal_g1_2m",
                "mcal_g2",
                "mcal_g2_1p",
                "mcal_g2_2p",
                "mcal_g2_1m",
                "mcal_g2_2m",
                "mcal_s2n",
                "mcal_T",
                "mcal_T_1p",
                "mcal_T_2p",
                "mcal_T_1m",
                "mcal_T_2m",
                "mcal_s2n_1p",
                "mcal_s2n_2p",
                "mcal_s2n_1m",
                "mcal_s2n_2m",
                "weight",
            ] + [f"mcal_mag_{b}" for b in bands]
        elif cat_type == "metadetect":
            # g1, g2, T, psf_g1, psf_g2, T, s2n, weight, magnitudes
            shear_cols = metadetect_variants(
                "g1",
                "g2",
                "T",
                "mcal_psf_g1",
                "mcal_psf_g2",
                "mcal_psf_T_mean",
                "s2n",
                "weight",
            )
            shear_cols += band_variants(
                bands, "mag", "mag_err", shear_catalog_type="metadetect"
            )
        else:
        
            shear_cols = [
                "dec",
                "psf_g1",
                "psf_g2",
                "g1",
                "g2",
                "psf_T_mean",
                "s2n",
                "T",
                "weight",
                "m",
            ] + [f"{shear_prefix}mag_{b}" for b in self.config["bands"]]

        shear_tomo_cols = ["bin"]

        if self.config["shear_catalog_type"] == "metacal":
            more_iters = ["shear_tomography_catalog", "response", ["R_gamma"]]
        else:
            more_iters = []

        it = self.combined_iterators(
            chunk_rows,
            "shear_catalog",
            "shear",
            shear_cols,
            "shear_tomography_catalog",
            "tomography",
            shear_tomo_cols,
            *more_iters,
        )
        
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
    
    def get_bin_edges(self, col):
        """
        Get the bin edges for a given column from the quantiles file
        """
        col = col.removeprefix(self.config["shear_prefix"])
        with self.open_input("shear_catalog_quantiles") as f:
            edges = f[f"quantiles/{col}"][:]
        return edges

    def plot_psf_shear(self):
        # mean shear in bins of PSF
        print("Making PSF shear plot")
        import matplotlib.pyplot as plt
        from scipy import stats
        
        psf_prefix = self.config["psf_prefix"]
        delta_gamma = self.config["delta_gamma"]

        psf_g_edges = self.get_bin_edges("psf_g1")
            
        p1 = MeanShearInBins(
            f"{psf_prefix}g1",
            psf_g_edges,
            delta_gamma,
            cut_source_bin=True,
            shear_catalog_type=self.config["shear_catalog_type"],
        )
        p2 = MeanShearInBins(
            f"{psf_prefix}g2",
            psf_g_edges,
            delta_gamma,
            cut_source_bin=True,
            shear_catalog_type=self.config["shear_catalog_type"],
        )

        psf_g_mid = 0.5 * (psf_g_edges[1:] + psf_g_edges[:-1])

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
       
        
        fig = self.open_output("g_psf_g", wrapper=True)
        
        # Include a small shift to be able to see the g1 / g2 points on the plot
        dx = 0.1 * (psf_g_edges[1]-psf_g_edges[0])
        idx = np.where(np.isfinite(mu1))[0]
        
        slope11, intercept11, mc_cov = fit_straight_line(mu1[idx], mean11[idx], std11[idx])
        std_err11 = mc_cov[0, 0] ** 0.5
        line11 = slope11 * (mu1) + intercept11
        
        slope12, intercept12, mc_cov = fit_straight_line(mu1[idx], mean12[idx], std12[idx])
        std_err12 = mc_cov[0, 0] ** 0.5
        line12 = slope12 * (mu1) + intercept12
        
        slope21, intercept21, mc_cov = fit_straight_line(mu2[idx], mean21[idx], std21[idx])
        std_err21 = mc_cov[0, 0] ** 0.5
        line21 = slope21 * (mu2) + intercept21
        
        slope22, intercept22, mc_cov = fit_straight_line(mu2[idx], mean22[idx], y_err=std22)
        std_err22 = mc_cov[0, 0] ** 0.5
        line22 = slope22 * (mu2) + intercept22
        
        plt.subplot(2, 1, 1)

        plt.plot(mu1, line11, color="red", label=r"$m=%.2e \pm %.2e$" % (slope11, std_err11))

        plt.plot(mu1, line12, color="blue", label=r"$m=%.2e \pm %.2e$" % (slope12, std_err12))
        plt.plot(mu1, [0] * len(line11), color="black")

        plt.errorbar(mu1 + dx, mean11, std11, label="g1", fmt="s", markersize=5, color="red")
        plt.errorbar(mu1 - dx, mean12, std12, label="g2", fmt="o", markersize=5, color="blue")
        
        plt.xlabel("PSF g1")
        plt.ylabel("Mean g")
        plt.legend()

        plt.subplot(2, 1, 2)

        plt.plot(mu2, line21, color="red", label=r"$m=%.2e \pm %.2e$" % (slope21, std_err21))
        plt.plot(mu2, line22, color="blue", label=r"$m=%.2e \pm %.2e$" % (slope22, std_err22))
        plt.plot(mu2, [0] * len(line22), color="black")
        
        plt.errorbar(mu2 + dx, mean21, std21, label="g1", fmt="s", markersize=5, color="red")
        plt.errorbar(mu2 - dx, mean22, std22, label="g2", fmt="o", markersize=5, color="blue")
        plt.xlabel("PSF g2")
        plt.ylabel("Mean g")
        plt.legend()
        plt.tight_layout()

        # This also saves the figure
        fig.close()
        
        f = self.open_output("g_psf_g_out")
        data   =[mu1,mu2,mean11,mean12,mean21,mean22,std11,std12,std21,std22,line11,line12,line21,line22]
        f.write(''.join([str(i) + '\n' for i in  data]))
        f.close()
    
    def plot_psf_size_shear(self):
        # mean shear in bins of PSF
        print("making shear psf size plot")
        import matplotlib.pyplot as plt
        from scipy import stats

        psf_prefix = self.config["psf_prefix"]
        delta_gamma = self.config["delta_gamma"]

        psf_T_edges = self.get_bin_edges("psf_T_mean")
        
        
        binnedShear = MeanShearInBins(
            f"{psf_prefix}T_mean",
            psf_T_edges,
            delta_gamma,
            cut_source_bin=True,
            shear_catalog_type=self.config["shear_catalog_type"],
            psf_unit_conv = self.config['psf_unit_conv']
        )
        
        while True:
            data = yield

            if data is None:
                break

            binnedShear.add_data(data)
            
        mu, mean1, mean2, std1, std2 = binnedShear.collect(self.comm)
        
        if self.rank != 0:
            return

        dx = 0.05 * (psf_T_edges[1] - psf_T_edges[0])
        idx = np.where(np.isfinite(mu))[0]
        slope1, intercept1, cov1 = fit_straight_line(mu[idx], mean1[idx], std1[idx])
        std_err1 = cov1[0, 0] ** 0.5
        line1 = slope1 * mu + intercept1
        slope2, intercept2, cov2 = fit_straight_line(mu[idx], mean2[idx], std2[idx])
        std_err2 = cov2[0, 0] ** 0.5
        line2 = slope2 * mu + intercept2
        
        fig = self.open_output("g_psf_T", wrapper=True)

        plt.plot(mu, line1, color="red", label=r"$m=%.2e \pm %.2e$" % (slope1, std_err1))
        plt.plot(mu, line2, color="blue", label=r"$m=%.2e \pm %.2e$" % (slope2, std_err2))
        plt.plot(mu, [0] * len(mu), color="black")
        plt.errorbar(mu + dx, mean1, std1, label="g1", fmt="s", markersize=5, color="red")
        plt.errorbar(mu - dx, mean2, std2, label="g2", fmt="o", markersize=5, color="blue")
        plt.xlabel("PSF $T$")
        plt.ylabel("Mean g")

        plt.legend(loc="best")
        plt.tight_layout()
        fig.close()
        
        f = self.open_output("g_psf_T_out")
        data   =[mu,mean1,mean2,std1,std2,line1,line2]
        f.write(''.join([str(i) + '\n' for i in  data]))
        f.close()
       
    
    def plot_snr_shear(self):
        # mean shear in bins of snr
        print("Making mean shear SNR plot")
        import matplotlib.pyplot as plt
        from scipy import stats

        # Parameters of the binning in SNR
        shear_prefix = self.config["shear_prefix"]
        delta_gamma = self.config["delta_gamma"]
    
        snr_edges = self.get_bin_edges("s2n")
            
        # This class includes all the cutting and calibration, both for
        # estimator and selection biases
        binnedShear = MeanShearInBins(
            f"{shear_prefix}s2n",
            snr_edges,
            delta_gamma,
            cut_source_bin=True,
            shear_catalog_type=self.config["shear_catalog_type"],
        )
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
        
        # Get the error on the mean
        dx = 0.05 * (snr_edges[1] - snr_edges[0])
        idx = np.where(np.isfinite(mu))[0]
        slope1, intercept1, mc_cov = fit_straight_line(mu[idx], mean1[idx], std1[idx])
        std_err1 = mc_cov[0, 0] ** 0.5
        line1 = slope1 * mu + intercept1
        
        slope2, intercept2, mc_cov = fit_straight_line(mu[idx], mean2[idx], std2[idx])
        std_err2 = mc_cov[0, 0] ** 0.5
        line2 = slope2 * mu + intercept2
        
        
        fig = self.open_output("g_snr", wrapper=True)

        plt.plot(mu, line1, color="red", label=r"$m=%.2e \pm %.2e$" % (slope1, std_err1))
        plt.plot(mu, line2, color="blue", label=r"$m=%.2e \pm %.2e$" % (slope2, std_err2))
        plt.plot(mu, [0] * len(mu), color="black")
        plt.errorbar(mu + dx, mean1, std1, label="g1", fmt="s", markersize=5, color="red")
        plt.errorbar(mu - dx, mean2, std2, label="g2", fmt="o", markersize=5, color="blue")
        plt.xlabel("SNR")
        plt.ylabel("Mean g")
        plt.legend()
        plt.tight_layout()
        fig.close()
        
        f = self.open_output("g_snr_out")
        data   =[mu,mean1,mean2,std1,std2,line1,line2]
        f.write(''.join([str(i) + '\n' for i in  data]))
        f.close()
    
    def plot_size_shear(self):
        # mean shear in bins of galaxy size
        print("Making mean shear galaxy size plot")
        import matplotlib.pyplot as plt
        from scipy import stats

        shear_prefix = self.config["shear_prefix"]
        delta_gamma = self.config["delta_gamma"]

        T_edges = self.get_bin_edges("T")
                
        binnedShear = MeanShearInBins(
            f"{shear_prefix}T",
            T_edges,
            delta_gamma,
            cut_source_bin=True,
            shear_catalog_type=self.config["shear_catalog_type"],
            psf_unit_conv = self.config['psf_unit_conv']
        )

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

        dx = 0.05 * (T_edges[1] - T_edges[0])
        idx = np.where(np.isfinite(mu))[0]
        slope1, intercept1, mc_cov = fit_straight_line(mu[idx], mean1[idx], y_err=std1[idx])
        std_err1 = mc_cov[0, 0] ** 0.5
        line1 = slope1 * mu + intercept1
        
        slope2, intercept2, mc_cov = fit_straight_line(mu[idx], mean2[idx], y_err=std2[idx])
        std_err2 = mc_cov[0, 0] ** 0.5
        line2 = slope2 * mu + intercept2
        
        fig = self.open_output("g_T", wrapper=True)

        plt.plot(mu, line1, color="red", label=r"$m=%.2e \pm %.2e$" % (slope1, std_err1))
        plt.plot(mu, line2, color="blue", label=r"$m=%.2e \pm %.2e$" % (slope2, std_err2))
        plt.plot(mu, [0] * len(mu), color="black")
        plt.errorbar(mu + dx, mean1, std1, label="g1", fmt="s", markersize=5, color="red")
        plt.errorbar(mu - dx, mean2, std2, label="g2", fmt="o",markersize=5, color="blue")
        
        plt.xlabel("galaxy size T")
        plt.ylabel("Mean g")
        plt.legend()
        plt.tight_layout()
        fig.close()
        
        f = self.open_output("g_T_out")
        data   =[mu,mean1,mean2,std1,std2,line1,line2]
        f.write(''.join([str(i) + '\n' for i in  data]))
        f.close()
        
    def plot_mag_shear(self):
        # mean shear in bins of magnitude
        print("Making mean shear band magnitude plot")
        import matplotlib.pyplot as plt
        from scipy import stats

        shear_prefix = self.config["shear_prefix"]
        delta_gamma = self.config["delta_gamma"]
        nbins = self.config["nbins"]
        
        stat = {}
        binnedShear = {}
        for band in self.config["bands"]:
            m_edges = self.get_bin_edges(f"{shear_prefix}mag_{band}")

            binnedShear[f"{band}"] = MeanShearInBins(
                f"{shear_prefix}mag_{band}",
                m_edges,
                delta_gamma,
                cut_source_bin=True,
                shear_catalog_type=self.config["shear_catalog_type"],
            )
        
        while True:
            # This happens when we have loaded a new data chunk
            data = yield

            # Indicates the end of the data stream
            if data is None:
                break
            
            for band in self.config["bands"]:
                binnedShear[f"{band}"].add_data(data)
                
        for band in self.config["bands"]:
            stat[f"mu_{band}"], stat[f"mean1_{band}"], stat[f"mean2_{band}"], stat[f"std1_{band}"], stat[f"std2_{band}"] = binnedShear[f"{band}"].collect(self.comm)

        if self.rank != 0:
            return

        for band in self.config["bands"]:

            dx = 0.05 * (m_edges[1] - m_edges[0])
        
            idx = np.where(np.isfinite(stat[f"mu_{band}"]))[0]
        
            stat[f"slope1_{band}"], stat[f"intercept1_{band}"], stat[f"mc_cov_{band}"] = fit_straight_line(stat[f"mu_{band}"][idx],
                                                                                                           stat[f"mean1_{band}"][idx],
                                                                                                           y_err=stat[f"std1_{band}"][idx])
            stat[f"std_err1_{band}"] = stat[f"mc_cov_{band}"][0, 0] ** 0.5
            stat[f"line1_{band}"] = stat[f"slope1_{band}"] * stat[f"mu_{band}"] + stat[f"intercept1_{band}"]

            stat[f"slope2_{band}"], stat[f"intercept2_{band}"], stat[f"mc_cov_{band}"] = fit_straight_line(stat[f"mu_{band}"][idx],
                                                                                                           stat[f"mean2_{band}"][idx],
                                                                                                           y_err=stat[f"std2_{band}"][idx])
            stat[f"std_err2_{band}"] = stat[f"mc_cov_{band}"][0, 0] ** 0.5
            stat[f"line2_{band}"] = stat[f"slope2_{band}"] * stat[f"mu_{band}"] + stat[f"intercept2_{band}"]

        fig = self.open_output("g_colormag", wrapper=True)
        for band,clr1,clr2 in zip(self.config["bands"],['maroon','firebrick','red'],['darkblue','royalblue','deepskyblue']):
            plt.subplot(2, 1, 1)
            plt.plot(stat[f"mu_{band}"], stat[f"line1_{band}"], color=clr1,
                     label=r"$m=%.2e \pm %.2e$" % (stat[f"slope1_{band}"], stat[f"std_err1_{band}"]))
            plt.plot(stat[f"mu_{band}"], [0] * len(stat[f"mu_{band}"]), color="black")
            plt.errorbar(stat[f"mu_{band}"] + dx, stat[f"mean1_{band}"], stat[f"std1_{band}"],
                         label=f"{band}-band", fmt="s", markersize=5, color=clr1)
            plt.ylabel("Mean g1")
            plt.xlabel("magnitude")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(stat[f"mu_{band}"], stat[f"line2_{band}"], color=clr2,
                     label=r"$m=%.2e \pm %.2e$" % (stat[f"slope2_{band}"], stat[f"std_err2_{band}"]))
            plt.plot(stat[f"mu_{band}"], [0] * len(stat[f"mu_{band}"]), color="black")
            plt.errorbar(stat[f"mu_{band}"] - dx, stat[f"mean2_{band}"], stat[f"std2_{band}"],
                         label=f"{band}-band", fmt="o",markersize=5, color=clr2)
            plt.ylabel("Mean g2")
            plt.xlabel("magnitude") 
            plt.legend()
        plt.tight_layout()
        fig.close()
        
    def plot_g_histogram(self):
        print("plotting histogram")
        import matplotlib.pyplot as plt
        from scipy import stats

        cat_type = self.config["shear_catalog_type"]
        delta_gamma = self.config["delta_gamma"]
        bins = 20
        edges = np.linspace(-1, 1, bins + 1)
        mids = 0.5 * (edges[1:] + edges[:-1])
        width = edges[1:] - edges[:-1]

        # Calibrate everything in the 2D bin
        _, cal = Calibrator.load(self.get_input("shear_tomography_catalog"))
        H1 = ParallelHistogram(edges)
        H2 = ParallelHistogram(edges)
        H1_weighted = ParallelHistogram(edges)
        H2_weighted = ParallelHistogram(edges)

        while True:
            data = yield

            if data is None:
                break

            qual_cut = data["bin"] != -1

            if cat_type == "metacal":
                g1 = data["mcal_g1"]
                g2 = data["mcal_g2"]
                w = data["weight"]
            elif cat_type == "metadetect":
                g1 = data["00/g1"]
                g2 = data["00/g2"]
                w = data["00/weight"]
            elif cat_type == "lensfit":
                dec = data["dec"]
                g1 = data["g1"]
                g2 = data["g2"]
                w = data["weight"]
            else:
                g1 = data["g1"]
                g2 = data["g2"]
                c1 = data['c1']
                c2 = data['c2']
                w = data["weight"]

            if cat_type=='metacal' or cat_type=='metadetect':
                g1, g2 = cal.apply(g1,g2)
                
            elif cat_type=='lensfit':
                # In KiDS, the additive bias is calculated and removed per North and South field
                # therefore, we add dec to split data into these fields. 
                # You can choose not to by setting dec_cut = 90 in the config, for example.
                g1, g2 = cal.apply(dec, g1,g2)
            else:
                g1, g2 = cal.apply(g1,g2,c1,c2)

            H1.add_data(g1)
            H2.add_data(g2)
            H1_weighted.add_data(g1, w)
            H2_weighted.add_data(g2, w)

        count1 = H1.collect(self.comm)
        count2 = H2.collect(self.comm)
        weight1 = H1_weighted.collect(self.comm)
        weight2 = H2_weighted.collect(self.comm)

        if self.rank != 0:
            return

        for i, count, weight in [(1, count1, weight1), (2, count2, weight2)]:
            with self.open_output(f"g{i}_hist", wrapper=True) as fig:
                plt.bar(
                    mids,
                    count,
                    width=width,
                    align="center",
                    color="lightblue",
                    label="Unweighted",
                )
                plt.bar(
                    mids,
                    weight,
                    width=width,
                    align="center",
                    color="none",
                    edgecolor="red",
                    label="Weighted",
                )
                plt.xlabel(f"g{i}")
                plt.ylabel("Count")
                plt.ylim(0, 1.1 * max(count1))
                plt.legend()

    def plot_snr_histogram(self):
        print("plotting snr histogram")
        import matplotlib.pyplot as plt

        delta_gamma = self.config["delta_gamma"]
        shear_prefix = self.config["shear_prefix"]
        bins = 10
        edges = np.logspace(1, 3, bins + 1)
        mids = 0.5 * (edges[1:] + edges[:-1])
        calc1 = ParallelMeanVariance(bins)

        while True:
            data = yield

            if data is None:
                break

            qual_cut = data["bin"] != -1

            b1 = np.digitize(data[f"{shear_prefix}s2n"][qual_cut], edges) - 1

            for i in range(bins):
                w = np.where(b1 == i)
                # Do more things here to establish
                calc1.add_data(i, data[f"{shear_prefix}s2n"][qual_cut][w])

        count1, mean1, var1 = calc1.collect(self.comm, mode="gather")
        if self.rank != 0:
            return
        std1 = np.sqrt(var1 / count1)
        fig = self.open_output("source_snr_hist", wrapper=True)
        plt.bar(
            mids,
            count1,
            width=edges[1:] - edges[:-1],
            edgecolor="black",
            align="center",
            color="blue",
        )
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.xscale("log")
        plt.xlabel("log(snr)")
        plt.ylabel(r"$N_{galaxies}$")
        plt.ylim(0, 1.1 * max(count1))
        fig.close()

    def plot_response_histograms(self):
        import matplotlib.pyplot as plt

        if self.comm:
            import mpi4py.MPI

        size = 10

        # This seems to be a reasonable range, though there are samples
        # with extremely high values
        edges = np.linspace(-3, 3, size + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        width = edges[1] - edges[0]
        cat_type = self.config["shear_catalog_type"]

        if cat_type == "metacal":
            # count of objects
            counts = np.zeros((2, 2, size))
            # make a separate histogram of the shear-sample-selected
            # objects
            counts_s = np.zeros((2, 2, size))
        else:
            # count of objects
            counts = np.zeros((size))
            # make a separate histogram of the shear-sample-selected
            # objects
            counts_s = np.zeros((size))

        # Main loop
        while True:
            data = yield

            if data is None:
                break

            if cat_type == "metadetect":
                # No per-object R values in metadetect
                continue

            # check if selected for any source bin
            in_shear_sample = data["bin"] != -1
            if cat_type == "metacal":
                B = np.digitize(data["R_gamma"], edges) - 1
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

            elif cat_type == "lensfit":
                B = np.digitize(data["m"], edges) - 1
                # loop through this chunk of data.
                for s, b in zip(in_shear_sample, B):
                    if (b >= 0) and (b < size):
                        counts[b] += 1
                        if s:
                            counts_s[b] += 1
            else:
                B = np.digitize(data["R"], edges) - 1
                # loop through this chunk of data.
                for s, b in zip(in_shear_sample, B):
                    if (b >= 0) and (b < size):
                        counts[b] += 1
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

        fig = self.open_output("response_hist", wrapper=True, figsize=(10, 5))

        plt.subplot(1, 2, 1)
        if cat_type == "metacal":
            manual_step_histogram(edges, counts[0, 0], label="R00", color="#1f77b4")
            manual_step_histogram(edges, counts[1, 1], label="R11", color="#ff7f0e")
            manual_step_histogram(edges, counts[0, 1], label="R01", color="#2ca02c")
            manual_step_histogram(edges, counts[1, 0], label="R10", color="#d62728")
        elif cat_type == "metadetect":
            plt.text(
                0.5,
                0.5,
                "This plot is intentionally left blank",
                horizontalalignment="center",
            )
            counts[:] = 0.909
        else:
            manual_step_histogram(edges, counts, label="R", color="#1f77b4")
        plt.ylim(0, counts.max() * 1.1)
        plt.xlabel("Response")
        plt.ylabel("Count")
        plt.title("All flag=0")

        plt.subplot(1, 2, 2)
        if cat_type == "metacal":
            manual_step_histogram(edges, counts_s[0, 0], label="R00", color="#1f77b4")
            manual_step_histogram(edges, counts_s[1, 1], label="R11", color="#ff7f0e")
            manual_step_histogram(edges, counts_s[0, 1], label="R01", color="#2ca02c")
            manual_step_histogram(edges, counts_s[1, 0], label="R10", color="#d62728")
        elif cat_type == "metadetect":
            plt.text(
                0.5,
                0.5,
                "This plot is intentionally left blank",
                horizontalalignment="center",
            )
            counts_s[:] = 0.909
        else:
            manual_step_histogram(edges, counts_s, label="R", color="#1f77b4")

        plt.ylim(0, counts_s.max() * 1.1)
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
        prefix = self.config["shear_prefix"]
        import matplotlib.pyplot as plt

        size = 10
        mag_min = 20
        mag_max = 30
        edges = np.linspace(mag_min, mag_max, size + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        width = edges[1] - edges[0]
        bands = self.config["bands"]
        shear_prefix = self.config["shear_prefix"]
        nband = len(bands)
        full_hists = [np.zeros(size, dtype=int) for b in bands]
        source_hists = [np.zeros(size, dtype=int) for b in bands]

        while True:
            data = yield

            if data is None:
                break

            for (b, h1, h2) in zip(bands, full_hists, source_hists):
                b1 = np.digitize(data[f"{shear_prefix}mag_{b}"], edges) - 1

                for i in range(size):
                    w = b1 == i
                    count = w.sum()
                    h1[i] += count

                    w &= data["bin"] >= 0
                    count = w.sum()
                    h2[i] += count

        if self.comm is not None:
            full_hists = reduce(self.comm, full_hists)
            source_hists = reduce(self.comm, source_hists)

        if self.rank == 0:
            fig = self.open_output(
                "source_mag_hist", wrapper=True, figsize=(4, nband * 3)
            )
            for i, (b, h1, h2) in enumerate(zip(bands, full_hists, source_hists)):
                plt.subplot(nband, 1, i + 1)
                plt.bar(
                    mid, h1, width=width, fill=False, label="Complete", edgecolor="r"
                )
                plt.bar(mid, h2, width=width, fill=True, label="WL Source", color="g")
                plt.xlabel(f"Mag {b}")
                plt.ylabel("N")
                if i == 0:
                    plt.legend()
            plt.tight_layout()
            fig.close()

class TXLensDiagnosticPlots(PipelineStage):
    """
    Make diagnostic plots of the lens catalog

    Currently this consists only of histograms of SNR and mag.
    """

    name = "TXLensDiagnosticPlots"
    # This tells ceci to launch under dask:
    dask_parallel = True

    inputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("lens_snr_hist", PNGFile),
        ("lens_mag_hist", PNGFile),
    ]

    config_options = {
        "block_size": 0,
        "delta_gamma": 0.02,
        "mag_min": 18,
        "mag_max": 28,
        "snr_min": 5,
        "snr_max": 200,
        "bands": "ugrizy",
    }

    def run(self):
        # PSF tests
        import matplotlib

        matplotlib.use("agg")

        files, data, nbin = self.load_data()
        self.plot_snr_histograms(data, nbin)
        self.plot_mag_histograms(data, nbin)

        # These need to stay open until the end
        for f in files:
            f.close()

    def load_data(self):
        _, da = import_dask()

        bands = self.config["bands"]
        # These need to stay open until dask has finished with them.:
        f = self.open_input("photometry_catalog")
        g = self.open_input("lens_tomography_catalog")

        # read nbin from metadata
        nbin = g["tomography"].attrs["nbin"]

        # Default to the automatic value but expose as an option
        block = self.config["block_size"]
        if block == 0:
            block = "auto"

        # Load data columns, lazily for dask
        data = {}
        for b in bands:
            data[f"mag_{b}"] = da.from_array(f[f"photometry/mag_{b}"], block)
            # all the blocks must be the same size. If the columns are of different
            # types then dask can sometimes select different block sizes for each column
            # which can cause problems. So we force them to be the same size.
            if block == "auto":
                block = data[f"mag_{b}"].chunksize
            data[f"snr_{b}"] = da.from_array(f[f"photometry/snr_{b}"], block)
        data["bin"] = da.from_array(g["tomography/bin"], block)

        # Avoid recomputing selections in each histogram by doing it externally here
        data["sel"] = da.nonzero(data["bin"] >= 0)
        for i in range(nbin):
            data[f"sel{i}"] = da.nonzero(data["bin"] == i)

        # Return the open files so they stay open until dask has finished with them.
        # and also the dict of lazy columns and the bin info
        return [f, g], data, nbin

    def plot_snr_histograms(self, data, nbin):
        smin = self.config["snr_min"]
        smax = self.config["snr_max"]
        bins = np.geomspace(smin, smax, 50)
        xlog = True
        self.plot_histograms(data, nbin, "snr", xlog, bins)

    def plot_mag_histograms(self, data, nbin):

        # Histogram ranges are read from configuration choices
        mmin = self.config["mag_min"]
        mmax = self.config["mag_max"]
        # let's do this in 0.5 mag bins
        bins = np.arange(mmin, mmax + 1e-6, 0.5)
        xlog = False

        self.plot_histograms(data, nbin, "mag", xlog, bins)

    def plot_histograms(self, data, nbin, name, xlog, bins):
        dask, da = import_dask()
        import matplotlib.pyplot as plt

        bands = self.config["bands"]
        nband = len(bands)

        hists = {}
        # Do a different set of histograms for each band
        for i, b in enumerate(bands):
            sel = data["sel"]
            # first do the global non-tomographic version (all selected objects)
            hists[b, -1] = da.histogram(data[f"{name}_{b}"][sel], bins=bins)
            # and also loop through tomo bins
            for j in range(nbin):
                sel = data[f"sel{j}"]
                hists[b, j] = da.histogram(data[f"{name}_{b}"][sel], bins=bins)

        # Launch actual dask computations so we have the data ready to plot
        # Doing these all at once seems to be faster
        print(f"Beginning {name} histogram compute")
        (hists,) = dask.compute(hists)
        print("Done")

        # Now make all the panels. The open_output method returns an object that
        # can be used as a context manager and will be saved automatically at the end to the
        # right location
        figsize = (4 * nband, 4 * (nbin + 1))
        with self.open_output(
            f"lens_{name}_hist", wrapper=True, figsize=figsize
        ) as fig:
            axes = fig.file.subplots(nbin + 1, nband, squeeze=False)
            for i, b in enumerate(bands):
                for j in range(-1, nbin):
                    bj = j if j >= 0 else "2D"
                    heights, edges = hists[b, j]
                    ax = axes[j + 1, i]
                    # This is my function to plot a line histogram from pre-computed edges and heights
                    manual_step_histogram(edges, heights, ax=ax)
                    if xlog:
                        ax.set_xscale("log")
                    ax.set_xlabel(f"Bin {bj} {b}-{name}")
            axes[0, 0].set_ylabel("Count")
            fig.file.tight_layout()


def reduce(comm, H):
    H2 = []
    rank = comm.Get_rank()
    for h in H:
        if rank == 0:
            hsum = np.zeros_like(h)
        else:
            hsum = None
        comm.Reduce(h, hsum)
        H2.append(hsum)
    return H2
