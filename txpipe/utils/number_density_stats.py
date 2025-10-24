import numpy as np


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


class LensNumberDensityStats:
    def __init__(self, nbin_lens, comm=None):
        self.nbin_lens = nbin_lens
        self.comm = comm
        self.lens_counts = np.zeros(nbin_lens)
        self.lens_counts_2d = 0.0

    def add_data(self, lens_bin):
        for i in range(self.nbin_lens):
            n = (lens_bin == i).sum()
            self.lens_counts[i] += n
            # each bin contributes to the 2D case
            self.lens_counts_2d += n

    def collect(self):
        if self.comm is None:
            lens_counts = self.lens_counts
            lens_counts_2d = self.lens_counts_2d
        else:
            import mpi4py.MPI

            lens_counts = np.zeros_like(self.lens_counts)
            self.comm.Reduce(
                [self.lens_counts, mpi4py.MPI.DOUBLE],
                [lens_counts, mpi4py.MPI.DOUBLE],
                op=mpi4py.MPI.SUM,
                root=0,
            )

            lens_counts_2d = self.comm.reduce(self.lens_counts_2d, op=mpi4py.MPI.SUM, root=0)

        return lens_counts, lens_counts_2d
