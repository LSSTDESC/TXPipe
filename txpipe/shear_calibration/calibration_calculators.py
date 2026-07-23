import numpy as np
from .names import META_VARIANTS
from .calibrators import MetaCalibrator, LensfitCalibrator, HSCCalibrator, MetaDetectCalibrator, NullCalibrator, AnaCalibrator
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
    """The CalibrationCalculator subclasses are used to compute calibration
    and other statistics of weak lensing catalogs, potentially in parallel on
    chunks of data at a time.

    One CalibrationCalculator should be used for each tomographic bin.
    Or for any other sub-selection, like bins in magnitude or size for null tests.

    The life-cycle of a CalibrationCalculator is as follows:
    - Create a CalibrationCalculator for each object using a function that selects your WL
      sample from input dictionaries (or similar) of data.
    - Add chunks of data to each calculator using the add_data method, which applies the
      selection function to each chunk, accumulates statistics, and returns the index of the selected objects
    - When all data has been added, call the collect method to finalize the statistics and return the results

    The data that is added to the calculator should contain shear columns appropriate to
    the specific type of calibrationn used. For example, the metadetect calculator expects
    columns 00/g1, 00/g2, etc.

    The selection function does not need to know about all these variants. The calculator
    will wrap the data dictionary passed in in a special class that chooses variant
    columns when they are looked up. So your selection function can just ask for, e.g.,
    "T", "s2n", or "mag_r", and the calculator will make sure it gets the right variant of
    that column for each selection.

    The final results are in the form of a BinStats object, which contains:
    - source_count: the raw number of objects
    - N_eff: the effective number of objects, accounting for weights
    - mean_e: the mean ellipticity values in the bin
    - sigma_e: the mean ellipticity dispersion per component
    - sigma: the standard deviation for the mean e1 and e2 separately in the bin
    - calibrator: a Calibrator subclass instance that calibrates this bin

    The attributes below apply to most of the subclasses, but the MetadetectCalculator has
    different types for them because it has to keep track of 5 variants separately.
    In that case the scalar attributes below are replaced by arrays of length 5, one for each variant,
    and the shear_stats keeps track of the mean and std dev of all of the variants.

    Attributes
    ----------
    selector: function
        A function that takes a chunk of data and selects objects to be used for
        calibration from it.  This is supplied by the user when initializing the
        class, and is used to select objects from each chunk of data as it is
        added.

    count: int
        The total number of objects selected across all chunks of data added so far
    sum_weights: float
        The sum of the weights of the selected objects across all chunks of data added so far
    sum_sq_weights: float
        The sum of the squares of the weights of the selected objects across all chunks of data added so far
    shear_stats: ParallelMeanVariance
        An accumulator to calculate the mean and variance of the selected shears, in g1 and g2 separately.
    """
    def __init__(self, selector):
        from parallel_statistics import ParallelMeanVariance
        self.selector = selector
        self.count = 0
        self.sum_weights = 0
        self.sum_sq_weights = 0
        self.shear_stats = ParallelMeanVariance(size=2)


class MetacalCalculator(CalibrationCalculator):
    """Calibration and stats calculator for metacalibration catalogs.

    See the CalibrationCalculator class for the use and contents of this class.
    """

    def __init__(self, selector, delta_gamma, resp_mean_diag=False):
        """
        Initialize the Calibrator using the function you will use to select
        objects. That function should take at least one argument,
        the chunk of data to select on.  It should look up the original
        names of the columns to select on, without the metacal suffix.

        The MetacalCalculator will wrap the data passed to it so that
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
        resp_mean_diag: bool
            If True, the mean response is forced to be a scalar multiple of the identity matrix, as in DES-Y3.
            If False (the default), the full response matrix is used.
        """
        from parallel_statistics import ParallelMean
        super().__init__(selector)

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

        Returns
        ----------
        sel_00: array
            The indices of the objects selected from this chunk of data using the baseline selection
            (i.e. no shear applied)

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

        g1 = data_00["g1"]
        g2 = data_00["g2"]
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
        R00 = data_1p["g1"][sel_00] - data_1m["g1"][sel_00]
        R01 = data_2p["g1"][sel_00] - data_2m["g1"][sel_00]
        R10 = data_1p["g2"][sel_00] - data_1m["g2"][sel_00]
        R11 = data_2p["g2"][sel_00] - data_2m["g2"][sel_00]

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
        Finalize and sum up all the response values, and return a BinStats
        obejct that collections calibration and statistics.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value
        allgather: bool
            If True, the response values will be returned for all the processors.

        Returns
        -------
        bin_stats: BinStats
            An object containing the final calibration and statistics for this bin.
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
    """A calibration and stats calculator for metadetect catalogs.

    See the CalibrationCalculator class for the use and contents of this class,
    but note that the attributes of the class are different because we have to
    keep track of 5 variants separately, and the shear_stats keeps track of the
    mean and std dev of all of the variants.
    """

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
        # so there is no point calling the parent __init__, becuase it would
        # set up the attributes of this class with the wrong types (scalars instead of arrays).

        self.selector = selector
        self.counts = np.zeros(5, dtype=int)
        self.sum_weights = np.zeros(5, dtype=float)
        self.sum_sq_weights = np.zeros(5, dtype=float)
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
        Finalize and sum up all the response values, and return a BinStats
        obejct that collections calibration and statistics.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value
        allgather: bool
            If True, the response values will be returned for all the processors.

        Returns
        -------
        bin_stats: BinStats
            An object containing the final calibration and statistics for this bin.
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
                    return None

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
        Finalize and sum up all the response values, and return a BinStats
        obejct that collections calibration and statistics.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value
        allgather: bool
            If True, the response values will be returned for all the processors.

        Returns
        -------
        bin_stats: BinStats
            An object containing the final calibration and statistics for this bin.
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
        Finalize and sum up all the response values, and return a BinStats
        obejct that collections calibration and statistics.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value
        allgather: bool
            If True, the response values will be returned for all the processors.

        Returns
        -------
        bin_stats: BinStats
            An object containing the final calibration and statistics for this bin.
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
        self.sum_sq_weights += np.sum(w**2)
        self.shear_stats.add_data(0, g1[sel], w)
        self.shear_stats.add_data(1, g2[sel], w)


        return sel

    def collect(self, comm=None, allgather=False) -> BinStats:
        """
        Finalize and sum up all the response values, and return a BinStats
        obejct that collections calibration and statistics.

        In this case the BinStats calibrator will be a NullCalibrator that does not apply any calibration.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value
        allgather: bool
            If True, the response values will be returned for all the processors.

        Returns
        -------
        bin_stats: BinStats
            An object containing the final calibration and statistics for this bin.
        """
        # The total number of objects is just the
        # number from all the processes summed together.
        if comm is not None:
            if allgather:
                count = comm.allreduce(self.count)
                sum_weights = comm.allreduce(self.sum_weights)
                sum_weights_sq = comm.allreduce(self.sum_sq_weights)

            else:
                count = comm.reduce(self.count)
                sum_weights = comm.reduce(self.sum_weights)
                sum_weights_sq = comm.reduce(self.sum_sq_weights)
        else:
            count = self.count
            sum_weights = self.sum_weights
            sum_weights_sq = self.sum_sq_weights

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


class AnaCalCalculator(CalibrationCalculator):
    """Calibration and stats calculator for AnaCal catalogs

    See the CalibrationCalculator class for the use and contents of this class
    """

    def __init__(self, selector, delta_gamma):
        """
        Initialize the Calibrator using the funtion you will use to select
        objects. That function should take at least one argument,
        the chunk of data to select on. It should look up the original
        names of the columns to select on; the calculator wraps the data
        with _DataWrapper so the same call also picks up ±γ variants when
        needed (mask + s2n + size + zbin cuts all go through this single
        code path).

        The selector can take further *args and **kwargs, passed in when
        adding data.

        Parameters
        ----------
        selector: function
            Function that selects objects. Must apply every cut whose
            shear response contributes to the selection bias (mask, s2n,
            size, zbin) — the calculator no longer applies any cut of its
            own.
        delta_gamma: float
            Half-difference in applied g between the ±1 variants
            (each variant is at ±delta_gamma from baseline).
        """
        from parallel_statistics import ParallelMean
        super().__init__(selector)

        self.delta_gamma = delta_gamma
        # e_meas ≡ wsel · e is the observable.  All accumulators
        # use uniform weight=1 and per-source values are pre-multiplied by wsel
        # where the chain-rule term needs it, so each collect() returns a
        # plain sample mean ⟨wsel · X⟩ (never divided by ⟨wsel⟩).
        # Note that wsel is smooth detection weight and 96% of them are 1.
        # Ensemble shear estimator:
        #     γ = ⟨e_meas⟩ / R_total,  R_total = R_shape + R_detect + R_sel
        # 0: ⟨wsel · de1/dg1⟩, 1: ⟨wsel · de2/dg2⟩
        self.shape_response = ParallelMean(size=2)
        # 0: ⟨(dwsel/dg1)·e1⟩, 1: ⟨(dwsel/dg2)·e2⟩
        self.detect_response = ParallelMean(size=2)
        # 0..3: ⟨wsel · e · 𝟙[sel_±]⟩
        self.sel_response = ParallelMean(size=4)

    def add_data(self, data, *args, **kwargs):
        """Select objects from a new chunk of data and tally their responses

        Parameters
        ----------
        data: dict
            Dictionary of data columns to select on and add
        *args
            Positional arguments to be passed to the selection function
        **kwargs
            Keyword arguments to be passed to the selection function.

        Returns
        -------
        sel: array
            The indicies of the objects selected from this chunk of data
        """
        # Baseline + four shifted views on the same chunk. _DataWrapper
        # routes ``data_1p["s2n"] -> data["s2n_1p"]``,
        # ``data_1p["m00"] -> data["m00_1p"]``,
        # ``data_1p["zbin"] -> data["zbin_1p"]``, etc.
        # Columns without a matching variant (mask_value, e1, weight, ...)
        # fall back to baseline.
        data_00 = _DataWrapper(data, "")
        data_1p = _DataWrapper(data, "_1p")
        data_1m = _DataWrapper(data, "_1m")
        data_2p = _DataWrapper(data, "_2p")
        data_2m = _DataWrapper(data, "_2m")

        # A single selector call per variant applies mask + s2n + size +
        # zbin cuts uniformly.  No per-cut special casing lives here now.
        select = self.selector(data_00, *args, **kwargs)
        sel_1p = self.selector(data_1p, *args, **kwargs)
        sel_1m = self.selector(data_1m, *args, **kwargs)
        sel_2p = self.selector(data_2p, *args, **kwargs)
        sel_2m = self.selector(data_2m, *args, **kwargs)

        # Convention-A column plan (see TXIngestAnacal):
        #   e1, e2      – e_meas ≡ wsel · e_raw (pre-multiplied, feed treecorr)
        #   e1_raw, e2_raw – raw shape (needed for R_detect numerator)
        #   wsel        – raw wsel (needed for R_shape numerator, Neff, μ)
        #   weight      – uniform 1 (feed treecorr's weight_column)
        e1 = data_00["e1"]                # = wsel · e_raw
        e2 = data_00["e2"]
        e1_raw = data_00["e1_raw"]
        e2_raw = data_00["e2_raw"]
        wsel = data_00["wsel"]
        weight_dg1 = data_00["weight_dg1"]
        weight_dg2 = data_00["weight_dg2"]
        de1_dg1 = data_00["de1_dg1"]
        de2_dg2 = data_00["de2_dg2"]

        # Baseline slices.
        w_sel = wsel[select]
        e1_sel = e1[select]              # already wsel · e_raw
        e2_sel = e2[select]

        # Bookkeeping: raw count + wsel sums (used for Kish's Neff).
        self.count += e1_sel.size
        self.sum_weights += np.sum(w_sel)
        self.sum_sq_weights += np.sum(w_sel ** 2)

        # All accumulators use weight=1 (uniform).  Per-source values are
        # pre-multiplied by wsel where the chain-rule term needs it, so each
        # ParallelMean returns ⟨wsel · X⟩ over the fed sample.

        # R_shape numerator: ⟨wsel · de_raw/dg⟩ over the baseline sample.
        self.shape_response.add_data(0, w_sel * de1_dg1[select])
        self.shape_response.add_data(1, w_sel * de2_dg2[select])
        # R_detect numerator: ⟨(dwsel/dg) · e_raw⟩ over the baseline sample.
        self.detect_response.add_data(0, weight_dg1[select] * e1_raw[select])
        self.detect_response.add_data(1, weight_dg2[select] * e2_raw[select])
        # R_sel numerator: ⟨wsel · e_raw⟩ over each ±γ variant sample —
        # equivalently ⟨e_meas⟩ since e1, e2 are already pre-multiplied.
        # (Spin-2 argument: N_1p, N_1m, N_2p, N_2m converge in the large-N
        # ensemble, so per-subset means telescope cleanly under the finite
        # difference; residual bias ~ (δN/N)·⟨e⟩, zero in ensemble.)
        self.sel_response.add_data(0, e1[sel_1p])
        self.sel_response.add_data(1, e1[sel_1m])
        self.sel_response.add_data(2, e2[sel_2p])
        self.sel_response.add_data(3, e2[sel_2m])

        # μ numerator: mean of e_meas (already pre-multiplied) over baseline.
        self.shear_stats.add_data(0, e1_sel)
        self.shear_stats.add_data(1, e2_sel)

        return select

    def collect(self, comm=None, allgather=False) -> BinStats:
        """
        Finalize and sum up all the response values, and return a BinStats
        obejct that collections calibration and statistics.

        Parameters
        ----------
        comm: MPI Communicator
            If supplied, all processors response values will be combined together.
            All processes will return the same final value
        allgather: bool
            If True, the response values will be returned for all the processors.

        Returns
        -------
        bin_stats: BinStats
            An object containing the final calibration and statistics for this bin.
        """
        # One helper for every scalar accumulator; comm.reduce returns
        # None on non-root ranks, which we let propagate through Neff.
        def _reduce(x):
            if comm is None:
                return x
            return comm.allreduce(x) if allgather else comm.reduce(x)

        count = _reduce(self.count)
        sum_weights = _reduce(self.sum_weights)
        sum_sq_weights = _reduce(self.sum_sq_weights)

        # Each ParallelXxx does its own MPI reduction inside .collect().
        mode = "allgather" if allgather else "gather"
        _, shape_means = self.shape_response.collect(comm, mode)   # (2,) means
        _, detect_means = self.detect_response.collect(comm, mode) # (2,) means
        _, sel_means = self.sel_response.collect(comm, mode)       # (4,) means
        _, mean_e, var_e = self.shear_stats.collect(comm, mode)

        # Convention A: every mean is a plain sample mean ⟨wsel · X⟩ over
        # the baseline (shape, detect, μ) or ±γ variant (sel) sample.
        # No ⟨wsel⟩ denominator anywhere — R_total already contains wsel
        # via the pre-multiplication in add_data.
        R_shape = 0.5 * (shape_means[0] + shape_means[1])
        R_detect = 0.5 * (detect_means[0] + detect_means[1])
        R_sel_1 = (sel_means[0] - sel_means[1]) / (2.0 * self.delta_gamma)
        R_sel_2 = (sel_means[2] - sel_means[3]) / (2.0 * self.delta_gamma)
        R_sel = 0.5 * (R_sel_1 + R_sel_2)
        R_total = R_shape + R_detect + R_sel

        Neff = None if sum_weights is None else sum_weights ** 2 / sum_sq_weights

        calibrator = AnaCalibrator(R_total, mean_e, mu_is_weighted=False)
        sigma_e = calibrator.calibrate_variance_to_sigma_e(var_e)
        sigma = calibrator.calibrate_sigma(np.sqrt(var_e))
        return BinStats(count, Neff, calibrator.mu, sigma_e, sigma, calibrator)
