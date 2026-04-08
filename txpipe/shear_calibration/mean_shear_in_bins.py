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
        self.g1 = ParallelMeanVariance(self.size)
        self.g2 = ParallelMeanVariance(self.size)
        self.x = ParallelMean(self.size)

        if shear_catalog_type == "metacal":
            self.calibrators = [MetacalCalculator(self.selector, delta_gamma) for i in range(self.size)]
        elif shear_catalog_type == "metadetect":
            self.calibrators = [MetaDetectCalculator(self.selector, delta_gamma) for i in range(self.size)]
        elif shear_catalog_type == "lensfit":
            print("for i in range ", self.size)
            self.calibrators = [LensfitCalculator(self.selector) for i in range(self.size)]
        elif shear_catalog_type == "hsc":
            self.calibrators = [HSCCalculator(self.selector) for i in range(self.size)]
        else:
            raise ValueError(f"Please specify metacal, metadetect, lensfit or hsc for shear_catalog in config.")

    def selector(self, data, i):
        x = data[self.x_name]
        if (self.shear_catalog_type == "lensfit") & (self.psf_unit_conv == True) & ("T" in self.x_name):
            pix2arcsec = 0.214
            x = x * pix2arcsec**2
        w = (x > self.limits[i]) & (x < self.limits[i + 1])
        if self.cut_source_bin:
            w &= data["bin"] != -1
        return np.where(w)[0]

    def add_data(self, data):
        for i in range(self.size):
            w = self.calibrators[i].add_data(data, i)
            if self.shear_catalog_type == "metacal":
                weight = data["weight"][w]
                self.g1.add_data(i, data["mcal_g1"][w], weight)
                self.g2.add_data(i, data["mcal_g2"][w], weight)
            elif self.shear_catalog_type == "metadetect":
                # The selector for metadetect returns the
                # selection for each of the variants. We just want
                # the first 00 selection here.
                w = w[0]
                weight = data["00/weight"][w]
                self.g1.add_data(i, data["00/g1"][w], weight)
                self.g2.add_data(i, data["00/g2"][w], weight)
            elif self.shear_catalog_type in ["lensfit", "metadetect"]:
                weight = data["weight"][w]
                self.g1.add_data(i, data["g1"][w], weight)
                self.g2.add_data(i, data["g2"][w], weight)
            elif self.shear_catalog_type == "hsc":
                weight = data["weight"][w]
                self.g1.add_data(i, data["g1"][w] - data["c1"][w], weight)
                self.g2.add_data(i, data["g2"][w] - data["c2"][w], weight)
            self.x.add_data(i, data[self.x_name][w], weight)

    def collect(self, comm=None):
        count1, g1, var1 = self.g1.collect(comm, mode="gather")
        count2, g2, var2 = self.g2.collect(comm, mode="gather")

        _, mu = self.x.collect(comm, mode="gather")
        # Now we have the complete sample we can get the calibration matrix
        # to apply to it.
        R = []
        K = []
        C_N = []
        C_S = []
        N = []
        Neff = []
        for i in range(self.size):
            if self.shear_catalog_type == "metacal":
                # Tell the Calibrators to work out the responses
                r, s, n, neff = self.calibrators[i].collect(comm)
                # and record the total (a 2x2 matrix)
                R.append(r + s)
                N.append(n)
                Neff.append(neff)
            elif self.shear_catalog_type == "metadetect":
                # Tell the Calibrators to work out the responses
                r, n, neff = self.calibrators[i].collect(comm)
                # and record the total (a 2x2 matrix)
                R.append(r)
                N.append(n)
                Neff.append(neff)
            elif self.shear_catalog_type == "lensfit":
                k, c_n, c_s, n, neff = self.calibrators[i].collect(comm)
                K.append(k)
                C_N.append(c_n)
                C_S.append(c_s)
                N.append(n)
                Neff.append(neff)
            else:
                r, k, n, neff = self.calibrators[i].collect(comm)
                K.append(k)
                R.append(r)
                N.append(n)
                Neff.append(neff)

        # Only the root processor does the rest
        if (comm is not None) and (comm.Get_rank() != 0):
            return None, None, None, None, None

        sigma1 = np.zeros(self.size)
        sigma2 = np.zeros(self.size)
        for i in range(self.size):
            # Get the shears and the errors on their means
            g = [g1[i], g2[i]]
            sigma = np.sqrt([var1[i] / Neff[i], var2[i] / Neff[i]])

            if self.shear_catalog_type in ["metacal", "metadetect"]:
                # Get the inverse response matrix to apply
                R_inv = np.linalg.inv(R[i])

                # Apply the matrix in full to the shears and errors
                g1[i], g2[i] = R_inv @ g
                sigma1[i], sigma2[i] = R_inv @ sigma
            elif self.shear_catalog_type == "lensfit":
                g1[i] = g1[i] * (1.0 / (1 + K[i]))
                g2[i] = g2[i] * (1.0 / (1 + K[i]))

                sigma1[i] = (1.0 / (1 + K[i])) * (sigma[0])
                sigma2[i] = (1.0 / (1 + K[i])) * (sigma[1])
            else:
                g1[i] = (g1[i] / (2 * R[i])) / (1 + K[i])
                g2[i] = (g2[i] / (2 * R[i])) / (1 + K[i])

                sigma1[i] = (sigma[0] / (2 * R[i])) / (1 + K[i])
                sigma2[i] = (sigma[1] / (2 * R[i])) / (1 + K[i])

        return mu, g1, g2, sigma1, sigma2
