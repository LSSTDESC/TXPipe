import numpy as np
from parallel_statistics import ParallelMeanVariance


class DepthMapperDR1:
    def __init__(self, pixel_scheme, snr_threshold, snr_delta, sparse=False, comm=None):
        """Class to build up depth maps iteratively as we cycle through a data set.

        Two methods should be used:
        - add_data which should be called each time a new data chunk is loaded
        - finalize, at the end to collect the results

        Parameters
        ----------
        pixel_scheme: PixelScheme object
            Converter from angle to pixel

        snr_threshold: float
            Value of SNR to use as the depth (e.g. 5.0 for 5 sigma depth)

        snr_delta: float, optional
            Half-width of the SNR values to use for the depth estimation

        sparse: bool, optional
            Whether to use sparse indexing for the calculation.  Faster if only a small number of pixels
            Are used.

        comm: MPI communicator, optional
            An MPI comm for parallel processing.  If None, calculation is serial.

        Returns
        -------

        pixel: array
            Indices of all pixels with any objects

        count: array
            Number of objects in each pixel

        depth: array
            Estimated depth of each pixel

        depth_var: array
            Estimated variance of depth of each pixel

        """
        self.pixel_scheme = pixel_scheme
        self.snr_threshold = snr_threshold
        self.snr_delta = snr_delta
        self.comm = comm
        self.sparse = sparse
        self.stats = ParallelMeanVariance(pixel_scheme.npix, sparse=sparse)

    def add_data(self, data):
        ra = data["ra"]
        dec = data["dec"]
        snr = data["snr"]
        mags = data["mag"]
        # Get healpix pixels
        pix_nums = self.pixel_scheme.ang2pix(ra, dec)

        # For each found pixel find all values hitting that pixel
        # and yield the index and their magnitudes
        for p in np.unique(pix_nums):
            mask = (pix_nums == p) & (abs(snr - self.snr_threshold) < self.snr_delta)
            self.stats.add_data(p, mags[mask])

    def finalize(self, comm=None):

        count, depth, depth_var = self.stats.collect(comm)

        # Generate the pixel indexing (if parallel and the master process) and
        # convert from sparse arrays to pixel, index arrays.if sparse
        if count is None:
            pixel = None
        elif self.sparse:
            pixel, count = count.to_arrays()
            _, depth = depth.to_arrays()
            _, depth_var = depth_var.to_arrays()
        else:
            pixel = np.arange(len(depth))

        return pixel, count, depth, depth_var


class BrightObjectMapper:
    def __init__(self, pixel_scheme, mag_threshold, sparse=False, comm=None):
        """Class to build up bright object maps iteratively as we cycle through a data set.

        Two methods should be used:
        - add_data which should be called each time a new data chunk is loaded
        - finalize, at the end to collect the results

        Parameters
        ----------
        pixel_scheme: PixelScheme object
            Converter from angle to pixel

        mag_threshold: float
            Value of magnitude to use as the cutoff for bright objects

        sparse: bool, optional
            Whether to use sparse indexing for the calculation.  Faster if only a small number of pixels
            Are used.

        comm: MPI communicator, optional
            An MPI comm for parallel processing.  If None, calculation is serial.

        Returns
        -------

        pixel: array
            Indices of all pixels with bright objects

        count: array
            Number of bright objects in each pixel

        brmag: array
            mean magnitude of bright objects in each pixel

        brmag_var: array
            variance of magnitude of bright objects in each pixel

        """
        self.pixel_scheme = pixel_scheme
        self.mag_threshold = mag_threshold
        self.comm = comm
        self.sparse = sparse
        self.stats = ParallelMeanVariance(pixel_scheme.npix, sparse=sparse)

    def add_data(self, data):
        ra = data["ra"]
        dec = data["dec"]
        ext = data["extendedness"]
        mags = data["mag"]
        # Get healpix pixels
        pix_nums = self.pixel_scheme.ang2pix(ra, dec)

        # For each found pixel find all values hitting that pixel
        # and yield the index and their magnitudes
        for p in np.unique(pix_nums):
            mask = (pix_nums == p) & (ext == 0) & (mags < self.mag_threshold)
            self.stats.add_data(p, mags[mask])

    def finalize(self, comm=None):

        count, brmag, brmag_var = self.stats.collect(comm)

        # Generate the pixel indexing (if parallel and the master process) and
        # convert from sparse arrays to pixel, index arrays.if sparse
        if count is None:
            pixel = None
        elif self.sparse:
            pixel, count = count.to_arrays()
            _, brmag = brmag.to_arrays()
            _, brmag_var = brmag_var.to_arrays()
        else:
            pixel = np.arange(len(brmag))

        return pixel, count, brmag, brmag_var
