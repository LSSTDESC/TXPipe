import numpy as np


class BinStats:
    """
    This is a small helper class to store and write the statistics of a
    single tomographic bin. It helps simplify some of the code below.
    """

    def __init__(self, source_count, N_eff, mean_e, sigma_e, calibrator):
        """
        Parameters
        ----------
        source_count: int
            The raw number of objects
        N_eff: int
            The effective number of objects
        mean_e: array or list
            Length 2. The mean ellipticity e1 and e2 in the bin
        sigma_e: float
            The ellipticity dispersion
        calibrator: Calibrator
            A Calibrator subclass instance that calibrates this bin
        """
        self.source_count = source_count
        self.N_eff = N_eff
        self.mean_e = mean_e
        self.sigma_e = sigma_e
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




class SourceNumberDensityStats:
    def __init__(self, nbin_source, shear_type, comm=None):
        from parallel_statistics import ParallelMeanVariance

        self.nbin_source = nbin_source
        self.comm = comm
        self.shear_type = shear_type
        self.shear_stats = [ParallelMeanVariance(2) for i in range(nbin_source)]
        self.shear_stats_2d = ParallelMeanVariance(2)

    def add_data(self, shear_data, shear_bin):
        for i in range(self.nbin_source):
            w = np.where(shear_bin == i)

            if self.shear_type == "metacal":
                self.shear_stats[i].add_data(0, shear_data["mcal_g1"][w], shear_data["weight"][w])
                self.shear_stats[i].add_data(1, shear_data["mcal_g2"][w], shear_data["weight"][w])
                # each bin contributes to the 2D
                self.shear_stats_2d.add_data(0, shear_data["mcal_g1"][w], shear_data["weight"][w])
                self.shear_stats_2d.add_data(1, shear_data["mcal_g2"][w], shear_data["weight"][w])
            elif self.shear_type == "metadetect":
                self.shear_stats[i].add_data(0, shear_data["00/g1"][w], shear_data["00/weight"][w])
                self.shear_stats[i].add_data(1, shear_data["00/g2"][w], shear_data["00/weight"][w])
                # each bin contributes to the 2D
                self.shear_stats_2d.add_data(0, shear_data["00/g1"][w], shear_data["00/weight"][w])
                self.shear_stats_2d.add_data(1, shear_data["00/g2"][w], shear_data["00/weight"][w])
            else:
                self.shear_stats[i].add_data(0, shear_data["g1"][w], shear_data["weight"][w])
                self.shear_stats[i].add_data(1, shear_data["g2"][w], shear_data["weight"][w])
                self.shear_stats_2d.add_data(0, shear_data["g1"][w], shear_data["weight"][w])
                self.shear_stats_2d.add_data(1, shear_data["g2"][w], shear_data["weight"][w])

    def collect(self):
        # Get the basic shear numbers - means, counts, variances
        nb = self.nbin_source

        # We have the nb bins plus a single non-tomographic bin
        # with everything in.
        variances = np.zeros((nb + 1, 2))
        means = np.zeros((nb + 1, 2))

        # The tomographic bins first
        for i in range(nb):
            _, means[i], variances[i] = self.shear_stats[i].collect(self.comm, mode="allgather")

        # and the 2D one
        _, means[nb], variances[nb] = self.shear_stats_2d.collect(self.comm, mode="allgather")
        return means, variances


