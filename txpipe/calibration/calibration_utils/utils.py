import numpy as np


class BinStats:
    """
    This is a small helper class to store and write the statistics of a
    single tomographic bin. It helps simplify some of the code below.
    """

    def __init__(self, source_count, N_eff, mean_e, sigma_e, sigma, calibrator):
        """
        Parameters
        ----------
        source_count: int
            The raw number of objects
        N_eff: float
            The effective number of objects
        mean_e: array or list
            Length 2. The mean ellipticity e1 and e2 in the bin
        sigma_e: float
            The ellipticity dispersion per component
        sigma: array or list
            Length 2. The standard deviation for the mean e1 and e2 in the bin
        calibrator: Calibrator
            A Calibrator subclass instance that calibrates this bin
        """
        self.source_count = source_count
        self.N_eff = N_eff
        self.mean_e = mean_e
        self.sigma_e = sigma_e
        self.sigma = sigma
        self.calibrator = calibrator

    def write_to(self, outfile, i):
        """
        Writes the bin statistics to an HDF5 file in the right place

        Parameters
        ----------
        outfile: an open HDF5 file object
            The output file
        i: int or str
            The index for this tomographic bin, or "2d"
        """
        group = outfile["counts"]
        if i == "2d":
            group["counts_2d"][:] = self.source_count
            group["N_eff_2d"][:] = self.N_eff
            # This might get saved by the calibrator also
            # but in case not we do it here.
            group["mean_e1_2d"][:] = self.mean_e[0]
            group["mean_e2_2d"][:] = self.mean_e[1]
            group["sigma_e_2d"][:] = self.sigma_e
        else:
            group["counts"][i] = self.source_count
            group["N_eff"][i] = self.N_eff
            group["mean_e1"][i] = self.mean_e[0]
            group["mean_e2"][i] = self.mean_e[1]
            group["sigma_e"][i] = self.sigma_e

        self.calibrator.save(outfile, i)
