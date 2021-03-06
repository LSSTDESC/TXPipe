import numpy as np
from parallel_statistics import ParallelMeanVariance, ParallelMean


def read_shear_catalog_type(stage):
    """
    Determine the type of shear catalog a stage is using as input.
    Returns a string, e.g. metacal, lensfit.
    Also sets shear_catalog_type in the stage's configuration
    so that it is available later and is saved in output.
    """
    with stage.open_input('shear_catalog', wrapper=True) as f:
        shear_catalog_type = f.catalog_type
        stage.config['shear_catalog_type'] = shear_catalog_type
    return shear_catalog_type

def metacal_variants(*names):
    return [
        name + suffix
        for suffix in ['', '_1p', '_1m', '_2p', '_2m']
        for name in names
    ]

def band_variants(bands, *names, shear_catalog_type='metacal'):
    if shear_catalog_type=='metacal':
        return [
            name + "_" + band + suffix
            for suffix in ['', '_1p', '_1m', '_2p', '_2m']
            for band in bands
            for name in names
        ]
    else:
        return [
            name + "_" + band
            for band in bands
            for name in names
        ]

def calculate_selection_response(g1, g2, sel_1p, sel_2p, sel_1m, sel_2m, delta_gamma):
    import numpy as np
    
    S = np.ones((2,2))
    S_11 = (g1[sel_1p].mean() - g1[sel_1m].mean()) / delta_gamma
    S_12 = (g1[sel_2p].mean() - g1[sel_2m].mean()) / delta_gamma
    S_21 = (g2[sel_1p].mean() - g2[sel_1m].mean()) / delta_gamma
    S_22 = (g2[sel_2p].mean() - g2[sel_2m].mean()) / delta_gamma
    
    # Also save the selection biases as a matrix.
    S[0,0] = S_11
    S[0,1] = S_12
    S[1,0] = S_21
    S[1,1] = S_22
    
    return S

def calculate_shear_response(g1_1p,g1_2p,g1_1m,g1_2m,g2_1p,g2_2p,g2_1m,g2_2m,delta_gamma):
    import numpy as np 
    
    n = len(g1_1p)
    R =  R = np.zeros((n,2,2))
    R_11 = (g1_1p - g1_1m) / delta_gamma
    R_12 = (g1_2p - g1_2m) / delta_gamma
    R_21 = (g2_1p - g2_1m) / delta_gamma
    R_22 = (g2_2p - g2_2m) / delta_gamma
    
    R[:,0,0] = R_11
    R[:,0,1] = R_12
    R[:,1,0] = R_21
    R[:,1,1] = R_22
    
    R = np.mean(R, axis=0)
    return R

def apply_metacal_response(R, S, g1, g2):
    # The values of R are assumed to already
    # have had appropriate weights included
    from numpy.linalg import pinv
    import numpy as np
    
    mcal_g = np.stack([g1,g2], axis=1)
    
    R_total = R+S
    
    # Invert the responsivity matrix
    Rinv = pinv(R_total)
    mcal_g = (Rinv @ mcal_g.T)
    
    return mcal_g[0], mcal_g[1]


def apply_lensfit_calibration(g1, g2, weight, c1=0, c2=0, sigma_e=0, m=0):
    w_tot = np.sum(weight)
    m = np.sum(weight*m)/w_tot        #if m not provided, default is m=0, so one_plus_K=1
    one_plus_K = 1.+m
    R = 1. - np.sum(weight*sigma_e)/w_tot
    g1 = (1./(one_plus_K))*((g1/R)-c1)       
    g2 = (1./(one_plus_K))*((g2/R)-c2)
    return g1, g2, weight, one_plus_K




class _DataWrapper:
    """
    This little helper class wraps dictionaries
    or other mappings, and whenever something is
    looked up in it, it first checks if there is
    a column with the specified suffix present instead
    and returns that if so.
    """
    def __init__(self, data, suffix):
        """Create 

        Parameters
        ----------
        data: dict

        suffix: str
        """
        self.suffix = suffix
        self.data = data

    def __getitem__(self, name):
        variant_name  = name + self.suffix
        if variant_name in self.data:
            return self.data[variant_name]
        else:
            return self.data[name]

    def __contains__(self, name):
        return (name in self.data)

class ParallelCalibratorMetacal:
    """
    This class builds up the total response and selection calibration
    factors for Metacalibration from each chunk of data it is given.
    At the end an MPI communuicator can be supplied to collect together
    the results from the different processes.

    To do this we need the function used to select the data, and the instance
    this function to each of the metacalibrated variants automatically by
    wrapping the data object passed in to it and modifying the names of columns
    that are looked up.
    """
    def __init__(self, selector, delta_gamma):
        """
        Initialize the Calibrator using the function you will use to select
        objects. That function should take at least one argument,
        the chunk of data to select on.  It should look up the original
        names of the columns to select on, without the metacal suffix.

        The ParallelCalibrator will then wrap the data passed to it so that
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
        self.selector = selector
        self.count = 0
        self.delta_gamma = delta_gamma
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
        data_00 = _DataWrapper(data, '')
        data_1p = _DataWrapper(data, '_1p')
        data_1m = _DataWrapper(data, '_1m')
        data_2p = _DataWrapper(data, '_2p')
        data_2m = _DataWrapper(data, '_2m')

        # These are the selections from this chunk of data
        # that we would make under different shears, the baseline
        # and all the others.  self.selector is a function that the user
        # supplied in init, not a method
        sel_00 = self.selector(data_00, *args, **kwargs)
        sel_1p = self.selector(data_1p, *args, **kwargs)
        sel_1m = self.selector(data_1m, *args, **kwargs)
        sel_2p = self.selector(data_2p, *args, **kwargs)
        sel_2m = self.selector(data_2m, *args, **kwargs)

        g1 = data_00['mcal_g1']
        g2 = data_00['mcal_g2']
        weight = data_00['weight']
        n = g1[sel_00].size

        # Record the count for this chunk, for summation later
        self.count += n

        # This is the estimator response, correcting  bias of the shear estimator itself
        # We have four components, and want the weighted mean of each, which we use
        # the ParallelMean class to get
        w00 = weight[sel_00]
        R00 = data_1p['mcal_g1'][sel_00] - data_1m['mcal_g1'][sel_00]
        R01 = data_2p['mcal_g1'][sel_00] - data_2m['mcal_g1'][sel_00]
        R10 = data_1p['mcal_g2'][sel_00] - data_1m['mcal_g2'][sel_00]
        R11 = data_2p['mcal_g2'][sel_00] - data_2m['mcal_g2'][sel_00]

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

        self.sel_bias_means.add_data(0, g1[sel_1p], weight[sel_1p])
        self.sel_bias_means.add_data(1, g1[sel_1m], weight[sel_1m])
        self.sel_bias_means.add_data(2, g1[sel_2p], weight[sel_2p])
        self.sel_bias_means.add_data(3, g1[sel_2m], weight[sel_2m])
        self.sel_bias_means.add_data(4, g2[sel_1p], weight[sel_1p])
        self.sel_bias_means.add_data(5, g2[sel_1m], weight[sel_1m])
        self.sel_bias_means.add_data(6, g2[sel_2p], weight[sel_2p])
        self.sel_bias_means.add_data(7, g2[sel_2m], weight[sel_2m])

        # The user of this class may need the base selection, so return it
        return sel_00

    def collect(self, comm=None):
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
        """
        # collect all the things we need
        if comm is not None:
            count = comm.allreduce(self.count)
        else:
            count = self.count


        # Collect the mean values we need
        _, S = self.sel_bias_means.collect(comm)
        _, R = self.cal_bias_means.collect(comm)

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
        S_mean = np.zeros((2,2))
        S_mean[0, 0] = S[0] - S[1]
        S_mean[0, 1] = S[2] - S[3]
        S_mean[1, 0] = S[4] - S[5]
        S_mean[1, 1] = S[6] - S[7]
        S_mean /= self.delta_gamma

        return R_mean, S_mean, count



class ParallelCalibratorNonMetacal:
    """
    This class builds up the total response calibration
    factors for NonMetacalibration shears from each chunk of data it is given.
    At the end an MPI communicator can be supplied to collect together
    the results from the different processes.

    To do this we need the function used to select the data, and the instance
    this function to each of the metacalibrated variants automatically by
    wrapping the data object passed in to it and modifying the names of columns
    that are looked up.
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
        self.selector = selector
        self.R = []
        self.K = []
        self.C = []
        self.counts = []

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
        data_00 = _DataWrapper(data, '')

        sel_00 = self.selector(data_00, *args, **kwargs)

        g1 = data_00['g1']
        g2 = data_00['g2']

        # Selector can return several reasonable ways to choose
        # objects - where result, boolean mask, integer indices
        if isinstance(sel_00, tuple):
            # tupe returned from np.where
            n = len(sel_00[0])
        elif np.issubdtype(sel_00.dtype, np.integer):
            # integer array
            n = len(sel_00)
        elif np.issubdtype(sel_00.dtype, np.bool_):
            # boolean selection
            n = sel_00.sum()
        else:
            raise ValueError("Selection function passed to Calibrator return type not known")
        w_tot = np.sum(data_00['weight'])
        m = np.sum(data_00['weight']*data_00['m'])/w_tot        #if m not provided, default is m=0, so one_plus_K=1
        K = 1.+m
        R = 1. - np.sum(data_00['weight']*data_00['sigma_e'])/w_tot
        C = np.stack([data_00['c1'],data_00['c2']],axis=1)

        self.R.append(R)
        self.K.append(K)
        self.C.append(C.mean(axis=0))
        self.counts.append(n)

        return sel_00

    def collect(self, comm=None):
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

        """
        # MPI allgather to get full arrays for everyone
        if comm is not None:
            self.R = sum(comm.allgather(self.R), [])
            self.K = sum(comm.allgather(self.K), [])
            self.C = sum(comm.allgather(self.C), [])
            self.counts = sum(comm.allgather(self.counts), [])

        R_sum = 0
        K_sum = 0
        C_sum = np.zeros((1,2))
        N = 0

        # Find the correctly weighted averages of all the values we have
        for R, K, C, n in zip(self.R, self.K, self.C, self.counts):
            # This deals with cases where n is 0 and R/S are NaN
            if n == 0:
                continue
            R_sum += R*n
            K_sum += K*n
            C_sum += C*n
            N += n

        if N == 0:
            R = np.nan
            K = np.nan
        else:
            R = R_sum / N
            K = K_sum / N

        C = C_sum / N
        
        return R, K, C, N 


class MeanShearInBins:
    def __init__(self, x_name, limits, delta_gamma, cut_source_bin=False, shear_catalog_type='metacal'):
        self.x_name = x_name
        self.limits = limits
        self.delta_gamma = delta_gamma
        self.cut_source_bin = cut_source_bin
        self.shear_catalog_type = shear_catalog_type
        self.size = len(self.limits) - 1

        # We have to work out the mean g1, g2 
        self.g1 = ParallelMeanVariance(self.size)
        self.g2 = ParallelMeanVariance(self.size)
        self.x  = ParallelMean(self.size)

        if shear_catalog_type=='metacal':
            self.calibrators = [ParallelCalibratorMetacal(self.selector, delta_gamma) for i in range(self.size)]
        else:
            self.calibrators = [ParallelCalibratorNonMetacal(self.selector) for i in range(self.size)]


    def selector(self, data, i):
        x = data[self.x_name]
        w = (x > self.limits[i]) & (x < self.limits[i+1])
        if self.cut_source_bin:
            w &= data['source_bin'] !=-1
        return np.where(w)


    def add_data(self, data):
        for i in range(self.size):
            w = self.calibrators[i].add_data(data, i)
            weight = data['weight'][w]
            if self.shear_catalog_type=='metacal':
                self.g1.add_data(i, data['mcal_g1'][w], weight)
                self.g2.add_data(i, data['mcal_g2'][w], weight)
            else:
                self.g1.add_data(i, data['g1'][w], weight)
                self.g2.add_data(i, data['g2'][w], weight)
            self.x.add_data(i, data[self.x_name][w], weight)

    def collect(self, comm=None):
        count1, g1, var1 = self.g1.collect(comm, mode='gather')
        count2, g2, var2 = self.g2.collect(comm, mode='gather')
        _, mu = self.x.collect(comm, mode='gather')

        # Now we have the complete sample we can get the calibration matrix
        # to apply to it.
        R = []
        K = []
        C =[]
        for i in range(self.size):
            if self.shear_catalog_type=='metacal':
                # Tell the Calibrators to work out the responses
                r, s, _ = self.calibrators[i].collect(comm)
                # and record the total (a 2x2 matrix)
                R.append(r+s)
            else:
                r, k, c, _ = self.calibrators[i].collect(comm)
                R.append(r)
                K.append(k)
                C.append(c)

        # Only the root processor does the rest
        if (comm is not None) and (comm.Get_rank() != 0):
            return None, None, None, None, None

        sigma1 = np.zeros(self.size)
        sigma2 = np.zeros(self.size)

        for i in range(self.size):
            # Get the shears and the errors on their means
            g = [g1[i], g2[i]]
            sigma = np.sqrt([var1[i]/count1[i], var2[i]/count2[i]])
            
            if self.shear_catalog_type=='metacal':
                # Get the inverse response matrix to apply
                R_inv = np.linalg.inv(R[i])

                # Apply the matrix in full to the shears and errors
                g1[i], g2[i] = R_inv @ g
                sigma1[i], sigma2[i] = R_inv @ sigma
            else:
                g1[i] = (1./(1+K[i]))*((g1[i]/R[i])-C[i][0][0])       
                g2[i] = (1./(1+K[i]))*((g2[i]/R[i])-C[i][0][1])
                sigma1[i] = (1./(1+K[i]))*((sigma[0]/R[i])-C[i][0][0])       
                sigma2[i] = (1./(1+K[i]))*((sigma[1]/R[i])-C[i][0][1])


        return mu, g1, g2, sigma1, sigma2

