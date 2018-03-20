import numpy as np


class ParallelStatsCalculator:
    def __init__(self, size, sparse=False):
        self.size = size
        self.sparse = sparse
        if sparse:
            import scipy.sparse
            self.mean = scipy.sparse.dok_matrix((size,1))
            self.count = scipy.sparse.dok_matrix((size,1))
            self.M2 = scipy.sparse.dok_matrix((size,1))
        else:
            self.mean = np.zeros((size,1))
            self.count = np.zeros((size,1))
            self.M2 = np.zeros((size,1))


    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def add_data(self, pixel, values):
        for value in values:
            self.count[pixel,0] += 1
            delta = value - self.mean[pixel,0]
            self.mean[pixel,0] += delta/self.count[pixel,0]
            delta2 = value - self.mean[pixel,0]
            self.M2[pixel,0] += delta * delta2

    # retrieve the mean and variance from an aggregate
    def finalize(self):
        if self.sparse:
            import scipy.sparse
            w = self.count.nonzero()
            self.variance = scipy.sparse.dok_matrix((self.size,1))
            self.variance[w,0] = self.M2[w].toarray()/(self.count[w].toarray() - 1)
        else:
            self.variance = self.M2/(self.count - 1)
            bad = self.count<2
            self.variance[bad] = np.nan            


    def calculate_serial(self, values_iterator):
        for pixel, values in values_iterator:
            self.add_data(pixel, values)

        self.finalize()
        return self.count, self.mean, self.variance

    def collect_dense(self, counts, means, variances):
        T =  np.zeros((self.size,1))
        S =  np.zeros((self.size,1))
        S2 = np.zeros((self.size,1))
        for count, mean, var in zip(counts, means, variances):
            # Deal with any NaNs
            mean[count==0] = 0
            var[count==0] = 0
            sums = mean*count
            sums2 = var*(count-1)
            T += count
            S += sums
            S2 += sums2
        variance = S2 / (T-1)
        mean = S / T
        count = T
        variance[count==0] = np.nan
        mean[count==0] = np.nan
        return count[:,0], mean[:,0], variance[:,0]


    def collect_sparse(self, counts, means, variances):
        import scipy.sparse
        T = scipy.sparse.dok_matrix((self.size,1))
        S = scipy.sparse.dok_matrix((self.size,1))
        S2 = scipy.sparse.dok_matrix((self.size,1))

        for count, mean, var in zip(counts, means, variances):
            # Deal with any NaNs
            w = count.nonzero()
            T += count
            S[w] += mean[w].toarray()*count[w].toarray()
            S2[w] += var[w].toarray() * (count[w].toarray() - 1)

        w = T.nonzero()
        variance = scipy.sparse.dok_matrix((self.size,1))
        variance[w] = S2[w] / (T[w].toarray() - 1)
        mean = S / T
        count = T
        return count, mean, variance

    def collect(self, counts, means, variances):
        if self.sparse:
            return self.collect_sparse(counts, means, variances)
        else:
            return self.collect_dense(counts, means, variances)


    def calculate_parallel(self, parallel_values_iterator, comm):
        # Each processor calculates the values for its bits of data
        self.calculate_serial(parallel_values_iterator)
        counts = comm.gather(self.count)
        means = comm.gather(self.mean)
        variances = comm.gather(self.variance)


        if comm.Get_rank()==0:
            count, mean, variance = self.collect(counts, means, variances)
        else:
            variance = None
            mean = None
            count = None
        
        return count, mean, variance


    def calculate(self, values_iterator, comm=None):
        if comm is None:
            count, mean, variance = self.calculate_serial(values_iterator)
            return count[:,0], mean[:,0], variance[:,0]
        else:
            count, mean, variance = self.calculate_parallel(values_iterator, comm)
            return count, mean, variance




