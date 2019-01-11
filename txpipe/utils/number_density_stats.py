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
            ParallelStatsCalculator(1)
            for i in range(nbin_source)
        ]

        self.lens_counts = np.zeros(nbin_lens)


    def add_data(self, shear_data, shear_bin, R, lens_bin):
        for i in range(self.nbin_source):
            w = np.where(shear_bin==i)
            m1 = R[:,0,0] # R11, taking the mean for the bin, TODO check if that's what we want to do
            m2 = R[:,1,1] # R22
            m = 0.5*(m1+m2)
            self.shear_stats[i].add_data(0, shear_data['mcal_g'][:,0][w])
            self.shear_stats[i].add_data(1, shear_data['mcal_g'][:,1][w])
            self.mean_response[i].add_data(0, m[w])

        for i in range(self.nbin_lens):
            n = (lens_bin==i).sum()
            self.lens_counts[i] += n



    def collect(self):
        # Get the basic shear numbers - means, counts, variances
        sigma_e = np.zeros(self.nbin_source)
        N_source = np.zeros(self.nbin_source)

        for i in range(self.nbin_source):
            _, mean_r, _ = self.mean_response[i].collect(self.comm, mode='allgather')
            counts, means, variances = self.shear_stats[i].collect(self.comm, mode='allgather')
            sigma_e[i] = (0.5 * (variances[0]/mean_r**2 + variances[1]/mean_r**2))**0.5
            N_source[i] = counts[0]

        return sigma_e, N_source, self.lens_counts

