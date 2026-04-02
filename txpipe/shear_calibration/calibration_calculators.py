import numpy as np
from .names import META_VARIANTS
from .calibrators import MetaCalibrator, LensfitCalibrator, HSCCalibrator, MetaDetectCalibrator, NullCalibrator
from .utils import BinStats

class _DataWrapper:
    """
    This little helper class wraps dictionaries
    or other mappings, and whenever something is
    looked up in it, it first checks if there is
    a column with the specified suffix present instead
    and returns that if so.
    """

    def __init__(self, data, suffix="", prefix=""):
        """Create

        Parameters
        ----------
        data: dict

        suffix: str
        """
        self.suffix = suffix
        self.prefix = prefix
        self.data = data

    def __getitem__(self, name):
        variant_name = self.prefix + name + self.suffix
        if variant_name in self.data:
            return self.data[variant_name]
        else:
            return self.data[name]

    def __contains__(self, name):
        return name in self.data


class CalibrationCalculator:
    def __init__(self, selector):
        from parallel_statistics import ParallelMeanVariance
        self.selector = selector
        self.count = 0
        self.sum_weights = 0
        self.sum_sq_weights = 0
        self.shear_stats = ParallelMeanVariance(size=2)


class MetacalCalculator(CalibrationCalculator):
    """
    This class builds up the total response and selection calibration
    factors for Metacalibration from each chunk of data it is given.
    At the end an MPI communicator can be supplied to collect together
    the results from the different processes.

    To do this we need the function used to select the data, and the instance
    this function to each of the metacalibrated variants automatically by
    wrapping the data object passed in to it and modifying the names of columns
    that are looked up.
    """

    def __init__(self, selector, delta_gamma, resp_mean_diag=False):
        """
        Initialize the Calibrator using the function you will use to select
        objects. That function should take at least one argument,
        the chunk of data to select on.  It should look up the original
        names of the columns to select on, without the metacal suffix.

        The MetacalCalculator will then wrap the data passed to it so that
        when a metacalibrated column is used for selection then the appropriate
        variant column is selected instead.

        The selector can take further *args and **kwargs, passed in when adding
        data.

        Parameters
        ----------
        selector: function
            Function that selects objects
        delta_gamma: float
            The difference in applied g between 1p and 1m metacal variants
        """
        from parallel_statistics import ParallelMean
        super().__init__(selector)

        self.selector = selector
        self.delta_gamma = delta_gamma
        self.resp_mean_diag = resp_mean_diag
        self.cal_bias_means = ParallelMean(size=4)
        self.sel_bias_means = ParallelMean(size=8)

    def add_data(self, data, *args, **kwargs):
        """Select objects from a new chunk of data and tally their responses

        Parameters
        ----------
        data: dict
            Dictionary of data columns to select on and add
        *args
            Positional arguments to be passed to the selection function
        **kwargs
            Keyword arguments to be passed to the selection function

        """
        # These all wrap the catalog such that lookups find the variant
        # column if available.
        # For example, if I look up data_1p["x"] then it will check if
        # data["x_1p"] exists and return that if so.  Otherwise it will
        # fall back to "x"
        data_00 = _DataWrapper(data, "")
        data_1p = _DataWrapper(data, "_1p")
        data_1m = _DataWrapper(data, "_1m")
        data_2p = _DataWrapper(data, "_2p")
        data_2m = _DataWrapper(data, "_2m")

        # These are the selections from this chunk of data
        # that we would make under different shears, the baseline
        # and all the others.  self.selector is a function that the user
        # supplied in init, not a method
        sel_00 = self.selector(data_00, *args, **kwargs)
        sel_1p = self.selector(data_1p, *args, **kwargs)
        sel_1m = self.selector(data_1m, *args, **kwargs)
        sel_2p = self.selector(data_2p, *args, **kwargs)
        sel_2m = self.selector(data_2m, *args, **kwargs)

        g1 = data_00["mcal_g1"]
        g2 = data_00["mcal_g2"]
        weight = data_00["weight"]
        weight1p = data_1p["weight"]
        weight1m = data_1m["weight"]
        weight2p = data_2p["weight"]
        weight2m = data_2m["weight"]
        n = g1[sel_00].size

        # Record the count for this chunk, for summation later
        self.count += n
        self.sum_weights += np.sum(weight[sel_00])
        self.sum_sq_weights += np.sum(weight[sel_00] ** 2)

        # This is the estimator response, correcting  bias of the shear estimator itself
        # We have four components, and want the weighted mean of each, which we use
        # the ParallelMean class to get
        w00 = weight[sel_00]
        R00 = data_1p["mcal_g1"][sel_00] - data_1m["mcal_g1"][sel_00]
        R01 = data_2p["mcal_g1"][sel_00] - data_2m["mcal_g1"][sel_00]
        R10 = data_1p["mcal_g2"][sel_00] - data_1m["mcal_g2"][sel_00]
        R11 = data_2p["mcal_g2"][sel_00] - data_2m["mcal_g2"][sel_00]

        # TODO: if there is a weight per variant would we use that here?
        # Not currently used though.
        self.cal_bias_means.add_data(0, R00, w00)
        self.cal_bias_means.add_data(1, R01, w00)
        self.cal_bias_means.add_data(2, R10, w00)
        self.cal_bias_means.add_data(3, R11, w00)

        # Now we handle the selection bias.  This value is given by the
        # difference in the (weighted) means of the shears under two selections,
        # the positive and negative shear, divided by the applied shear
        # See the second term in equation 14 of https://arxiv.org/pdf/1702.02601
        # We need to calculate eight means:
        # (g1 or g2) X (shear applied to 1 or 2) X (plus or minus shear)
        # We again use the ParallelMean class to handle this for us.

        self.sel_bias_means.add_data(0, g1[sel_1p], weight1p[sel_1p])
        self.sel_bias_means.add_data(1, g1[sel_1m], weight1m[sel_1m])
        self.sel_bias_means.add_data(2, g1[sel_2p], weight2p[sel_2p])
        self.sel_bias_means.add_data(3, g1[sel_2m], weight2m[sel_2m])
        self.sel_bias_means.add_data(4, g2[sel_1p], weight1p[sel_1p])
        self.sel_bias_means.add_data(5, g2[sel_1m], weight1m[sel_1m])
        self.sel_bias_means.add_data(6, g2[sel_2p], weight2p[sel_2p])
        self.sel_bias_means.add_data(7, g2[sel_2m], weight2m[sel_2m])

        self.shear_stats.add_data(0, g1[sel_00], w00)
        self.shear_stats.add_data(1, g2[sel_00], w00)

        # The user of this class may need the base selection, so return it
        return sel_00

    def collect(self, comm=None, allgather=False) -> BinStats:
        """
        Finalize and sum up all the response values, returning separate
        R (estimator response) and S (selection bias) 2x2 matrices

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value

        Returns
        -------
        R: 2x2 array
            Estimator response matrix
        S: 2x2 array
            Selection bias matrix

        Neff: float
            Sum(weights)**2 / Sum(weights**2)
        """
        # collect all the things we need
        if comm is not None:
            if allgather:
                count = comm.allreduce(self.count)
                sum_weights = comm.allreduce(self.sum_weights)
                sum_sq_weights = comm.allreduce(self.sum_sq_weights)
            else:
                count = comm.reduce(self.count)
                sum_weights = comm.reduce(self.sum_weights)
                sum_sq_weights = comm.reduce(self.sum_sq_weights)
        else:
            count = self.count
            sum_weights = self.sum_weights
            sum_sq_weights = self.sum_sq_weights

        # Collect the mean values we need
        mode = "allgather" if allgather else "gather"
        _, S = self.sel_bias_means.collect(comm, mode)
        _, R = self.cal_bias_means.collect(comm, mode)
        _, mean_e, var_e = self.shear_stats.collect(comm, mode)

        # Unpack the flat mean R and S values into
        # matrices
        R_mean = np.zeros((2, 2))
        R_mean[0, 0] = R[0]
        R_mean[0, 1] = R[1]
        R_mean[1, 0] = R[2]
        R_mean[1, 1] = R[3]
        R_mean /= self.delta_gamma

        # Need to take all the diffs to compute the
        # S term
        S_mean = np.zeros((2, 2))
        S_mean[0, 0] = S[0] - S[1]
        S_mean[0, 1] = S[2] - S[3]
        S_mean[1, 0] = S[4] - S[5]
        S_mean[1, 1] = S[6] - S[7]
        S_mean /= self.delta_gamma

        if sum_weights is None:
            Neff = None
        else:
            Neff = sum_weights**2 / sum_sq_weights

        if self.resp_mean_diag:
            # Sets response to scalar R[0,0]==R[1,1] = (R[0,0]+R[1,1])/2 and nulls the off-diagonal elements (used in DES-Y3)
            Ravg = (R_mean[0, 0] + R_mean[1, 1]) / 2.0
            R_mean[1, 0] = R_mean[0, 1] = 0
            R_mean[0, 0] = R_mean[1, 1] = Ravg

            Savg = (S_mean[0, 0] + S_mean[1, 1]) / 2.0
            S_mean[1, 0] = S_mean[0, 1] = 0
            S_mean[0, 0] = S_mean[1, 1] = Savg

        calibrator = MetaCalibrator(R_mean, S_mean, mean_e, mu_is_calibrated=False)
        sigma_e = calibrator.calibrate_variance_to_sigma_e(var_e)
        sigma = calibrator.calibrate_sigma(np.sqrt(var_e))
        bin_stats = BinStats(count, Neff, calibrator.mu, sigma_e, sigma, calibrator)
        return bin_stats


class MetaDetectCalculator(CalibrationCalculator):
    """ """

    def __init__(self, selector, delta_gamma):
        """

        Parameters
        ----------
        selector: function
            Function that selects objects
        delta_gamma: float
            The difference in applied g between 1p and 1m metacal variants
        """
        from parallel_statistics import ParallelMean, ParallelMeanVariance
        # The MetaDetectCalculator is a bit different to the others in that
        # it has to keep track of 5 copies of everything, one for each variant.
        # so there is no point calling the parent __init__.

        self.selector = selector
        self.counts = np.zeros(5, dtype=int)
        self.sum_weights = np.zeros(5, dtype=int)
        self.sum_sq_weights = np.zeros(5, dtype=int)
        self.delta_gamma = delta_gamma
        self.shear_stats = ParallelMeanVariance(size=10)

    def add_data(self, data, *args, **kwargs):
        """Select objects from a new chunk of data and tally their responses

        Parameters
        ----------
        data: dict
            Dictionary of data columns to select on and add
        *args
            Positional arguments to be passed to the selection function
        **kwargs
            Keyword arguments to be passed to the selection function

        """
        selections = []
        prefixes = [m + "/" for m in META_VARIANTS]
        for i, p in enumerate(prefixes):
            data_p = _DataWrapper(data, prefix=p)
            sel = self.selector(data_p, *args, **kwargs)
            selections.append(sel)
            w = data_p["weight"][sel]
            if w.size == 0:
                continue
            g1 = data_p["g1"][sel]
            g2 = data_p["g2"][sel]
            self.shear_stats.add_data(2 * i + 0, g1, w)
            self.shear_stats.add_data(2 * i + 1, g2, w)
            self.counts[i] += w.size
            self.sum_weights[i] += np.sum(w)
            self.sum_sq_weights[i] += np.sum(w**2)

        return selections

    def collect(self, comm=None, allgather=False) -> BinStats:
        """
        Finalize and sum up all the response values, returning separate
        R (estimator response) 2x2 matrix
        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value
        Returns
        -------
        R: 2x2 array
            Estimator response matrix
        """
        # collect all the things we need
        mode = "allgather" if allgather else "gather"
        _, mean_e, var_e = self.shear_stats.collect(comm, mode)

        if comm is not None:
            if allgather:
                counts = comm.allreduce(self.counts)
                sum_weights = comm.allreduce(self.sum_weights)
                sum_sq_weights = comm.allreduce(self.sum_sq_weights)
            else:
                counts = comm.reduce(self.counts)
                sum_weights = comm.reduce(self.sum_weights)
                sum_sq_weights = comm.reduce(self.sum_sq_weights)

                if comm.rank > 0:
                    return None, None, None

        else:
            counts = self.counts
            sum_weights = self.sum_weights
            sum_sq_weights = self.sum_sq_weights

        # The ordering of these arrays is, from above:
        # 0: g1
        # 1: g2
        # 2: g1_1p
        # 3: g2_1p
        # 4: g1_1m
        # 5: g2_1m
        # 6: g1_2p
        # 7: g2_2p
        # 8: g1_2m
        # 9: g2_2m

        # Compute the mean R components
        R = np.zeros((2, 2))
        R[0, 0] = mean_e[2] - mean_e[4]  # g1_1p - g1_1m
        R[0, 1] = mean_e[6] - mean_e[8]  # g1_2p - g1_2m
        R[1, 0] = mean_e[3] - mean_e[5]  # g2_1p - g2_1m
        R[1, 1] = mean_e[7] - mean_e[9]  # g2_2p - g2_2m
        R /= self.delta_gamma

        Neff = sum_weights[0] ** 2 / sum_sq_weights[0]

        calibrator = MetaDetectCalibrator(R, mean_e[:2], mu_is_calibrated=False)
        mu = calibrator.apply(mean_e[0], mean_e[1], subtract_mean=False)
        sigma_e = calibrator.calibrate_variance_to_sigma_e(var_e[0:2])
        sigma = calibrator.calibrate_sigma(np.sqrt(var_e[:2]))
        bin_stats = BinStats(counts[0], Neff, mu, sigma_e, sigma, calibrator)

        # we just want the count of the 00 base catalog
        return bin_stats


class LensfitCalculator(CalibrationCalculator):
    """
    This class builds up the total calibration
    factors for lensfit-convention shears from each chunk of data it is given.
    Note here we derive the c-terms from the data (in constrast to averaging
    values derived from simulations and stored in the catalog.)
    At the end an MPI communicator can be supplied to collect together
    the results from the different processes.

    """

    def __init__(self, selector, dec_cut=True, input_m_is_weighted=False):
        """
        Initialize the Calibrator using the function you will use to select
        objects. That function should take at least one argument,
        the chunk of data to select on.

        The selector can take further *args and **kwargs, passed in when adding
        data.

        Parameters
        ----------
        selector: function
            Function that selects objects
        """
        from parallel_statistics import ParallelMean
        super().__init__(selector)
        self.selector = selector
        # Create a set of calculators that will calculate (in parallel)
        # the three quantities we need to compute the overall calibration
        # We create these, then add data to them below, then collect them
        # together over all the processes
        self.K = ParallelMean(1)
        self.C_N = ParallelMean(2)
        self.C_S = ParallelMean(2)
        # In KiDS, the additive bias is calculated and removed per North and South field
        # we have implemented a config to choose whether or not to do this split
        self.dec_cut = dec_cut
        self.input_m_is_weighted = input_m_is_weighted

    def add_data(self, data, *args, **kwargs):
        """Select objects from a new chunk of data and tally their responses

        Parameters
        ----------
        data: dict
            Dictionary of data columns to select on and add

        *args
            Positional arguments to be passed to the selection function
        **kwargs
            Keyword arguments to be passed to the selection function

        """
        # These all wrap the catalog such that lookups find the variant
        # column if available
        # This is just to let the selection tools access data.variant for feedback
        data = _DataWrapper(data, "")
        sel = self.selector(data, *args, **kwargs)

        # Extract the calibration quantities for the selected objects
        w = data["weight"]
        K = data["m"]
        g1 = data["g1"]
        g2 = data["g2"]
        dec = data["dec"]
        n = g1[sel].size

        # Record the count for this chunk, for summation later
        self.count += n
        self.sum_weights += np.sum(w[sel])
        self.sum_sq_weights += np.sum(w[sel] ** 2)

        self.shear_stats.add_data(0, g1[sel], w[sel])
        self.shear_stats.add_data(1, g2[sel], w[sel])

        # Accumulate the calibration quantities so that later we
        # can compute the weighted mean of the values
        if self.input_m_is_weighted:
            # if the m values are already weighted don't use the weights here
            self.K.add_data(0, K[sel])
        else:
            # if not apply the weights
            self.K.add_data(0, K[sel], w[sel])

        if self.dec_cut == True:
            Nmask = dec[sel] > -25.0
            Smask = dec[sel] <= -25.0

            self.C_N.add_data(0, g1[sel][Nmask], w[sel][Nmask])
            self.C_N.add_data(1, g2[sel][Nmask], w[sel][Nmask])
            self.C_S.add_data(0, g1[sel][Smask], w[sel][Smask])
            self.C_S.add_data(1, g2[sel][Smask], w[sel][Smask])
        else:

            self.C_N.add_data(0, g1[sel], w[sel])
            self.C_N.add_data(1, g2[sel], w[sel])
            self.C_S.add_data(0, np.zeros(n), np.zeros(n))
            self.C_S.add_data(1, np.zeros(n), np.zeros(n))

        return sel

    def collect(self, comm=None, allgather=False) -> BinStats:
        """
        Finalize and sum up all the response values, returning calibration
        quantities.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value

        Returns
        -------

        K: float
            K = (1+m) calibration

        C: float array
            c1, c2 additive biases (weighted average of g1 and g2)

        Neff: float
            Sum(weights)**2 / Sum(weights**2)

        """
        # The total number of objects is just the
        # number from all the processes summed together.
        if comm is not None:
            if allgather:
                count = comm.allreduce(self.count)
                sum_weights = comm.allreduce(self.sum_weights)
                sum_sq_weights = comm.allreduce(self.sum_sq_weights)
            else:
                count = comm.reduce(self.count)
                sum_weights = comm.reduce(self.sum_weights)
                sum_sq_weights = comm.reduce(self.sum_sq_weights)

        else:
            count = self.count
            sum_weights = self.sum_weights
            sum_sq_weights = self.sum_sq_weights

        # Collect the weighted means of these numbers.
        # this collects all the values from the different
        # processes and over all the chunks of data
        mode = "allgather" if allgather else "gather"
        _, K = self.K.collect(comm, mode)
        _, C_N = self.C_N.collect(comm, mode)
        _, C_S = self.C_S.collect(comm, mode)
        _, mean_e, var_e = self.shear_stats.collect(comm, mode)

        if sum_weights is None:
            Neff = None
        else:
            Neff = sum_weights**2 / sum_sq_weights

        calibrator = LensfitCalibrator(K[0], C_N, C_S, dec_cut=self.dec_cut)
        mu = calibrator.apply(mean_e[0], mean_e[1], subtract_mean=False)
        sigma_e = calibrator.calibrate_variance_to_sigma_e(var_e)
        sigma = calibrator.calibrate_sigma(np.sqrt(var_e))
        bin_stats = BinStats(count, Neff, mu, sigma_e, sigma, calibrator)

        return bin_stats


class HSCCalculator(CalibrationCalculator):
    """
    This class builds up the total response calibration
    factors for HSC-convention shear-calibration from each chunk of data it is
    given.
    At the end an MPI communicator can be supplied to collect together
    the results from the different processes.

    """

    def __init__(self, selector):
        """
        Initialize the Calibrator using the function you will use to select
        objects. That function should take at least one argument,
        the chunk of data to select on.

        The selector can take further *args and **kwargs, passed in when adding
        data.

        Parameters
        ----------
        selector: function
            Function that selects objects
        """
        from parallel_statistics import ParallelMean
        super().__init__(selector)
        # Create a set of calculators that will calculate (in parallel)
        # the three quantities we need to compute the overall calibration
        # We create these, then add data to them below, then collect them
        # together over all the processes
        self.K = ParallelMean(1)
        self.R = ParallelMean(1)

    def add_data(self, data, *args, **kwargs):
        """Select objects from a new chunk of data and tally their responses

        Parameters
        ----------
        data: dict
            Dictionary of data columns to select on and add

        *args
            Positional arguments to be passed to the selection function
        **kwargs
            Keyword arguments to be passed to the selection function

        """
        # These all wrap the catalog such that lookups find the variant
        # column if available

        # This is just to let the selection tools access data.variant for feedback
        data = _DataWrapper(data, "")
        sel = self.selector(data, *args, **kwargs)

        # Extract the calibration quantities for the selected objects
        w = data["weight"]
        K = data["m"]
        g1 = data['g1']
        g2 = data['g2']
        R = 1.0 - data["sigma_e"] ** 2
        n = w[sel].size
        self.count += n
        self.sum_weights += np.sum(w[sel])
        self.sum_sq_weights += np.sum(w[sel] ** 2)
        self.shear_stats.add_data(0, g1[sel] - data["c1"][sel], w[sel])
        self.shear_stats.add_data(1, g2[sel] - data["c2"][sel], w[sel])

        w = w[sel]

        # Accumulate the calibration quantities so that later we
        # can compute the weighted mean of the values
        self.R.add_data(0, R[sel], w)
        self.K.add_data(0, K[sel], w)
        return sel

    def collect(self, comm=None, allgather=False) -> BinStats:
        """
        Finalize and sum up all the response values, returning calibration
        quantities.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value

        Returns
        -------
        R: float
            R calibration factor

        K: float
            K = (1+m) calibration

        N: int
            Total object count

        Neff: float
            Total effective number of galaxies


        """
        # The total number of objects is just the
        # number from all the processes summed together.
        if comm is not None:
            if allgather:
                count = comm.allreduce(self.count)
                sum_weights = comm.allreduce(self.sum_weights)
                sum_sq_weights = comm.allreduce(self.sum_sq_weights)

            else:
                count = comm.reduce(self.count)
                sum_weights = comm.reduce(self.sum_weights)
                sum_sq_weights = comm.reduce(self.sum_sq_weights)
        else:
            count = self.count
            sum_weights = self.sum_weights
            sum_sq_weights = self.sum_sq_weights
        # Collect the weighted means of these numbers.
        # this collects all the values from the different
        # processes and over all the chunks of data
        mode = "allgather" if allgather else "gather"
        _, R = self.R.collect(comm, mode)
        _, K = self.K.collect(comm, mode)
        _, mean_e, var_e = self.shear_stats.collect(comm, mode)

        if sum_weights is None:
            Neff = None
        else:
            Neff = sum_weights**2 / sum_sq_weights

        calibrator = HSCCalibrator(R[0], K[0])
        mu = calibrator.apply(mean_e[0], mean_e[1], subtract_mean=False)

        sigma_e = calibrator.calibrate_variance_to_sigma_e(var_e)
        sigma = calibrator.calibrate_sigma(np.sqrt(var_e))
        bin_stats = BinStats(count, Neff, mu, sigma_e, sigma, calibrator)


        return bin_stats


class MockCalculator(CalibrationCalculator):
    """
    This class calculates tomographic statistics for mock catalogs
    where no calibration is necessary.

    It only accumulates the statistics of the selected object weights
    instead of any calibration quantities like its sibling classes.

    """

    def __init__(self, selector):
        """
        Initialize the Calibrator using the function you will use to select
        objects. That function should take at least one argument,
        the chunk of data to select on.

        The selector can take further *args and **kwargs, passed in when adding
        data.

        Parameters
        ----------
        selector: function
            Function that selects objects
        """
        super().__init__(selector)
        # There is nothing else to do for the mock calculator.

    def add_data(self, data, *args, **kwargs):
        """Select objects from a new chunk of data and tally their responses

        Parameters
        ----------
        data: dict
            Dictionary of data columns to select on and add

        *args
            Positional arguments to be passed to the selection function
        **kwargs
            Keyword arguments to be passed to the selection function

        """
        data = _DataWrapper(data, "")
        sel = self.selector(data, *args, **kwargs)

        # Extract the calibration quantities for the selected objects
        w = data["weight"]
        g1 = data['g1']
        g2 = data['g2']
        n = w[sel].size
        self.count += n
        w = w[sel]
        self.sum_weights += np.sum(w)
        self.sum_weights_sq += np.sum(w**2)
        self.shear_stats.add_data(0, g1[sel], w)
        self.shear_stats.add_data(1, g2[sel], w)


        return sel

    def collect(self, comm=None, allgather=False) -> BinStats:
        """
        Finalize and sum up all the response values, returning calibration
        quantities.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value

        Returns
        -------
        N: int
            Total object count

        Neff: float
            Total effective number of galaxies


        """
        # The total number of objects is just the
        # number from all the processes summed together.
        if comm is not None:
            if allgather:
                count = comm.allreduce(self.count)
                sum_weights = comm.allreduce(self.sum_weights)
                sum_weights_sq = comm.allreduce(self.sum_weights_sq)

            else:
                count = comm.reduce(self.count)
                sum_weights = comm.reduce(self.sum_weights)
                sum_weights_sq = comm.reduce(self.sum_weights_sq)
        else:
            count = self.count
            sum_weights = self.sum_weights
            sum_weights_sq = self.sum_weights_sq

        # Collect the weighted means of these numbers.
        # this collects all the values from the different
        # processes and over all the chunks of data
        mode = "allgather" if allgather else "gather"
        _, mean_e, var_e = self.shear_stats.collect(comm, mode)
        calibrator = NullCalibrator()

        Neff = sum_weights**2 / sum_weights_sq
        mu = calibrator.apply(mean_e[0], mean_e[1], subtract_mean=False)
        sigma_e = calibrator.calibrate_variance_to_sigma_e(var_e)
        sigma = calibrator.calibrate_sigma(np.sqrt(var_e))
        bin_stats = BinStats(count, Neff, mu, sigma_e, sigma, calibrator)

        return bin_stats


