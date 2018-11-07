from ..utils import choose_pixelization, HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator
import numpy as np

class Mapper:
    def __init__(self, pixel_scheme, lens_bins, source_bins, tasks=(0,1,2)):
        self.pixel_scheme = pixel_scheme
        self.source_bins = source_bins
        self.lens_bins = lens_bins
        self.tasks = tasks
        self.stats = {}
        for b in self.bins:
            for t in self.tasks:
                self.stats[(b,task)] = ParallelStatsCalculator(self.pixel_scheme.npix)

    def add_data(self, shear_data, bin_data, m_data):
        npix = self.pixel_scheme.npix

        # Get pixel indices
        pix_nums = pixel_scheme.ang2pix(shear_data['ra'], shear_data['dec'])

        # TODO: change from unit weights for lenses
        lens_weights = np.ones_like(shear_data['ra'])

        # In advance make the mask indicating which tomographic bin
        # Each galaxy is in.  Later we will AND this with the selection
        # for each pixel.
        masks_lens = [bin_data['lens_bin'] == b for b in self.lens_bins]
        masks_source = [bin_data['source_bin'] == b for b in self.source_bins]

        for p in np.unique(pix_nums):  # Loop through pixels
            if p < 0 or p >= npix:
                continue

            # All the data points that hit this pixel
            mask_pix = (pix_nums == p)

            # Number counts.
            t = 0
            if t in self.tasks:
                # Loop through the tomographic lens bins
                for i,b in enumerate(self.lens_bins):
                    mask_bins = masks_lens[i]
                    # Loop through tasks (number counts, gamma_x)
                    self.stats[(b,t)].add_data(p, w[mask_pix & mask_bins])

            # Shears
            for t in (1,2):
                # We may be skipping tasks in future
                if not t in self.tasks:
                    continue
                # Loop through tomographic source bins
                for i,b in enumerate(self.source_bins):
                    mask_bins = masks_source[i]
                    w = shear_data['mcal_g'][:, t - 1]
                    self.stats[(b,t)].add_data(p, w[mask_pix & mask_bins])


    def finalize(self, comm=None):
        ngal = {}
        g1 = {}
        g2 = {}

        # TODO: support sparse
        pixel = np.arange(self.pixel_scheme.npix)

        is_master = (comm is None) or (comm.Get_rank()==0)
        for t in self.tasks:
            for b in self.bins:
            stats = self.stats[(b,t)]
            count, mean, _ = stats.finalize(comm)

            if not is_master:
                continue

            count[np.isnan(count)] = 0
            mean[np.isnan(mean)] = 0

            count = count.reshape(self.pixel_scheme.shape)
            mean = mean.reshape(self.pixel_scheme.shape)

            if t==0:
                # In the case of the density bin the mean is actually
                # the mean weight.  So we want to multiply to get total
                # weighted galaxy count
                ngal[b] = mean * count
            elif t==1:
                g1[b] = mean
            elif t==2:
                g2[b] = mean
        return pixel, ngal, g1, g2


