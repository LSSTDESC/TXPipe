import numpy as np

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
