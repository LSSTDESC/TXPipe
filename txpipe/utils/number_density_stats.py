from .stats import ParallelStatsCalculator
import numpy as np


class SourceNumberDensityStats:
    def __init__(self, nbin_source, shear_type, comm=None):
        self.nbin_source = nbin_source
        self.comm = comm
        self.shear_type = shear_type
        self.shear_stats = [ParallelStatsCalculator(2) for i in range(nbin_source)]

    def add_data(self, shear_data, shear_bin):
        for i in range(self.nbin_source):
            w = np.where(shear_bin == i)
            if self.shear_type == 'metacal':
                self.shear_stats[i].add_data(0, shear_data['mcal_g1'][w])
                self.shear_stats[i].add_data(1, shear_data['mcal_g2'][w])
            else:
                self.shear_stats[i].add_data(0, shear_data['g1'][w])
                self.shear_stats[i].add_data(1, shear_data['g2'][w])

    def collect(self):
        # Get the basic shear numbers - means, counts, variances
        sigma_e = np.zeros(self.nbin_source)

        for i in range(self.nbin_source):
            _, _, variances = self.shear_stats[i].collect(self.comm, mode='allgather')

            # This needs to be divided by the response outside here,
            # as this value is not calibrated
            sigma_e[i] = (0.5 * (variances[0] + variances[1])) ** 0.5

        return sigma_e


class LensNumberDensityStats:
    def __init__(self, nbin_lens, comm=None):
        self.nbin_lens = nbin_lens
        self.comm = comm
        self.lens_counts = np.zeros(nbin_lens)

    def add_data(self, lens_bin):

        for i in range(self.nbin_lens):
            n = (lens_bin == i).sum()
            self.lens_counts[i] += n

    def collect(self):

        if self.comm is None:
            lens_counts = self.lens_counts
        else:
            import mpi4py.MPI

            lens_counts = np.zeros_like(self.lens_counts)
            self.comm.Reduce(
                [self.lens_counts, mpi4py.MPI.DOUBLE],
                [lens_counts, mpi4py.MPI.DOUBLE],
                op=mpi4py.MPI.SUM,
                root=0,
            )

        return lens_counts
