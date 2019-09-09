from .stats import ParallelStatsCalculator
import numpy as np

class NumberDensityStats:
    def __init__(self, nbin_source, nbin_lens, comm=None):
        self.nbin_source = nbin_source
        self.nbin_lens = nbin_lens
        self.comm = comm
        self.shear_stats = [
            ParallelStatsCalculator(2)
            for i in range(nbin_source)
        ]

        self.mean_response = [
            ParallelStatsCalculator(4)
            for i in range(nbin_source)
        ]

        self.lens_counts = np.zeros(nbin_lens)


    def add_data(self, shear_data, shear_bin, R, lens_bin):
        for i in range(self.nbin_source):
            w = np.where(shear_bin==i)
            R00 = R[:,0,0]
            R01 = R[:,0,1]
            R10 = R[:,1,0]
            R11 = R[:,1,1]
            self.shear_stats[i].add_data(0, shear_data['mcal_g1'][w])
            self.shear_stats[i].add_data(1, shear_data['mcal_g2'][w])
            self.mean_response[i].add_data(0, R00[w])
            self.mean_response[i].add_data(1, R01[w])
            self.mean_response[i].add_data(2, R10[w])
            self.mean_response[i].add_data(3, R11[w])

        for i in range(self.nbin_lens):
            n = (lens_bin==i).sum()
            self.lens_counts[i] += n



    def collect(self):
        # Get the basic shear numbers - means, counts, variances
        sigma_e = np.zeros(self.nbin_source)
        N_source = np.zeros(self.nbin_source)
        mean_R = np.zeros((self.nbin_source, 2, 2))

        for i in range(self.nbin_source):
            _, mean_r, _ = self.mean_response[i].collect(self.comm, mode='allgather')
            counts, means, variances = self.shear_stats[i].collect(self.comm, mode='allgather')
            mean_r0 = 0.5 * (mean_r[0] + mean_r[3]) # 0.5*(R00 + R11)
            # Improve this - not great
            sigma_e[i] = (0.5 * (variances[0]/mean_r0**2 + variances[1]/mean_r0**2))**0.5
            N_source[i] = counts[0]
            mean_R[i,0,0] = mean_r[0]
            mean_R[i,0,1] = mean_r[1]
            mean_R[i,1,0] = mean_r[2]
            mean_R[i,1,1] = mean_r[3]

        lc = np.zeros_like(self.lens_counts)
        if self.comm is not None:
            import mpi4py.MPI
            self.comm.Reduce(
                [self.lens_counts, mpi4py.MPI.DOUBLE],
                [lc, mpi4py.MPI.DOUBLE],
                op = mpi4py.MPI.SUM,
                root = 0
            )

        return sigma_e, mean_R, N_source, lc

