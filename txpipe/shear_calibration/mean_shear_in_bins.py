import numpy as np
from .calibration_calculators import MetacalCalculator, MetaDetectCalculator, LensfitCalculator, HSCCalculator

class MeanShearInBins:
    def __init__(
        self, x_name, limits, delta_gamma, cut_source_bin=False, shear_catalog_type="metacal", psf_unit_conv=False
    ):
        from parallel_statistics import ParallelMeanVariance, ParallelMean

        self.x_name = x_name
        self.limits = limits
        self.delta_gamma = delta_gamma
        self.cut_source_bin = cut_source_bin
        self.shear_catalog_type = shear_catalog_type
        self.psf_unit_conv = psf_unit_conv
        self.size = len(self.limits) - 1
        # We have to work out the mean g1, g2
        self.x = ParallelMean(self.size)

        if shear_catalog_type == "metacal":
            self.calibrators = [MetacalCalculator(self.selector, delta_gamma) for i in range(self.size)]
        elif shear_catalog_type == "metadetect":
            self.calibrators = [MetaDetectCalculator(self.selector, delta_gamma) for i in range(self.size)]
        elif shear_catalog_type == "lensfit":
            self.calibrators = [LensfitCalculator(self.selector, dec_cut=False) for i in range(self.size)]
        elif shear_catalog_type == "hsc":
            self.calibrators = [HSCCalculator(self.selector) for i in range(self.size)]
        else:
            raise ValueError(f"Please specify metacal, metadetect, lensfit or hsc for shear_catalog in config.")

    def selector(self, data, i):
        x = data[self.x_name]

        # Hack to convert from pixel^2 to arcsec^2 for KIDS sizes, which are in pixel^2, but the cuts are in arcsec^2.
        # We should fix this on ingestion
        if (self.shear_catalog_type == "lensfit") & (self.psf_unit_conv == True) & ("T" in self.x_name):
            pix2arcsec = 0.214
            x = x * pix2arcsec**2

        # select objects in bin i of the x variable.
        w = (x > self.limits[i]) & (x < self.limits[i + 1])

        # Optionally cut down to the source sample only
        if self.cut_source_bin:
            w &= data["bin"] != -1
        return np.where(w)[0]

    def add_data(self, data):
        for i in range(self.size):
            # The i argument to add_data is the argument that is passed
            # through to the "selector" method above.
            w = self.calibrators[i].add_data(data, i)
            if self.shear_catalog_type == "metadetect":
                # the metadetector selector returns selections
                # for all 5 variants. We just want the unsheared on
                # here.
                w = w[0]
                weight = data["00/weight"][w]
            else:
                weight = data["weight"][w]
    
            self.x.add_data(i, data[self.x_name][w], weight)

    def collect(self, comm=None):

        _, mu = self.x.collect(comm, mode="gather")
        g1 = np.zeros(self.size)
        g2 = np.zeros(self.size)
        sigma1 = np.zeros(self.size)
        sigma2 = np.zeros(self.size)

        for i in range(self.size):
            stats = self.calibrators[i].collect(comm, allgather=True)
            sigma1[i], sigma2[i] = stats.sigma / np.sqrt(stats.N_eff)
            g1[i], g2[i] = stats.mean_e

        return mu, g1, g2, sigma1, sigma2
