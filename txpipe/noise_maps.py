from .base_stage import PipelineStage
from .data_types import ShearCatalog, TomographyCatalog, MapsFile, \
                        NoiseMaps, HDFFile
import numpy as np
from .utils.mpi_utils import mpi_reduce_large

class TXNoiseMaps(PipelineStage):
    """
    Generate a suite of random noise maps by randomly
    rotating individual galaxy measurements.

    """
    name='TXNoiseMaps'

    
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('lens_tomography_catalog', TomographyCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        # We get the pixelization info from the diagnostic maps
        ('mask', MapsFile),
        ('lens_maps', MapsFile),
    ]

    outputs = [
        ('noise_maps', NoiseMaps),
    ]

    config_options = {
        'chunk_rows': 100000,
        'lensing_realizations': 30,
        'clustering_realizations': 1,
    }        

    def run(self):
        from .utils import choose_pixelization

        # get the number of bins.
        nbin_source, nbin_lens, ngal_maps, mask, map_info = self.read_inputs()
        
        pixel_scheme = choose_pixelization(**map_info)
        lensing_realizations = self.config['lensing_realizations']
        clustering_realizations = self.config['clustering_realizations']

        # The columns we will need
        shear_cols = ['ra', 'dec', 'weight', 'mcal_g1', 'mcal_g2']

        # Make the iterators
        chunk_rows = self.config['chunk_rows']

        it = self.combined_iterators(chunk_rows,
                'shear_catalog', 'shear', shear_cols,
                'shear_tomography_catalog','tomography', ['source_bin'],
                'lens_tomography_catalog','tomography', ['lens_bin'],
            )

        # Get a mapping from healpix indices to masked pixel indices
        # This reduces memory usage.  We could use a healsparse array
        # here, but I'm not sure how to do that best with our
        # many realizations.  Possiby a recarray?
        index_map = np.zeros(pixel_scheme.npix, dtype=np.int64) - 1
        c = 0
        for i in range(pixel_scheme.npix):
            if mask[i] > 0:
                index_map[i] = c
                c += 1

        # Number of unmasked pixels
        npix = c

        if self.rank == 0:
            nmaps = nbin_source * (2 * lensing_realizations + 1) + nbin_lens * clustering_realizations * 2
            nGB = (npix * nmaps * 8) / 1000.**3
            print(f"Allocating maps of size {nGB:.2f} GB")

        # lensing g1, g2
        G1 = np.zeros((npix, nbin_source, lensing_realizations))
        G2 = np.zeros((npix, nbin_source, lensing_realizations))
        # lensing weight
        GW = np.zeros((npix, nbin_source))
        # clustering map - n_gal to start with
        ngal_split = np.zeros((npix, nbin_lens, clustering_realizations, 2), dtype=np.int32)
        # TODO: Clustering weights go here



        # Loop through the data
        for (s, e, data) in it:
            print(f"Rank {self.rank} processing rows {s} - {e}")
            source_bin = data['source_bin']
            lens_bin = data['lens_bin']
            ra = data['ra']
            dec = data['dec']
            orig_pixels = pixel_scheme.ang2pix(ra, dec)
            pixels = index_map[orig_pixels]
            n = e - s

            w = data['weight']
            g1 = data['mcal_g1'] * w
            g2 = data['mcal_g2'] * w

            # randomly select a half for each object
            split = np.random.binomial(1, 0.5, (n, clustering_realizations))

            # random rotations of the g1, g2 values
            phi = np.random.uniform(0, 2*np.pi, (n, lensing_realizations))
            c = np.cos(phi)
            s = np.sin(phi)
            g1r =  c * g1[:, np.newaxis] + s * g2[:, np.newaxis]
            g2r = -s * g1[:, np.newaxis] + c * g2[:, np.newaxis]

            for i in range(n):
                # convert to the index in the partial space
                pix = pixels[i]

                if pix < 0:
                    continue

                sb = source_bin[i]
                lb = lens_bin[i]
                # build up the rotated map for each bin
                if sb >= 0:
                    G1[pix, sb, :] += g1r[i]
                    G2[pix, sb, :] += g2r[i]
                    GW[pix, sb] += w[i]
                # Build up the ngal for the random half for each bin
                for j in range(clustering_realizations):
                    if lb >= 0:
                        ngal_split[pix, lb, j, split[i]] += 1
                    # TODO add to clustering weight too


        # Sum everything at root
        if self.comm is not None:
            mpi_reduce_large(G1, self.comm)
            mpi_reduce_large(G2, self.comm)
            mpi_reduce_large(GW, self.comm)
            mpi_reduce_large(ngal_split, self.comm)
            if self.rank != 0:
                del G1, G2, GW, ngal_split


        if self.rank==0:
            print("Saving maps")
            outfile = self.open_output('noise_maps', wrapper=True)

            # The top section has the metadata in
            group = outfile.file.create_group("maps")
            group.attrs['nbin_source'] = nbin_source
            group.attrs['lensing_realizations'] = lensing_realizations
            group.attrs['clustering_realizations'] = clustering_realizations

            metadata = {**self.config, **map_info}

            pixels = np.where(mask>0)[0]

            for b in range(nbin_source):
                for i in range(lensing_realizations):

                    bin_mask = np.where(GW[:, b]>0)

                    g1 = G1[:, b, i] / GW[:, b]
                    g2 = G2[:, b, i] / GW[:, b]

                    outfile.write_map(f"rotation_{i}/g1_{b}", 
                        pixels[bin_mask], g1[bin_mask], metadata)

                    outfile.write_map(f"rotation_{i}/g2_{b}", 
                        pixels[bin_mask], g2[bin_mask], metadata)

            for b in range(nbin_lens):
                for i in range(clustering_realizations):
                    # We have computed the first half already,
                    # and we have the total map from an earlier stage
                    half1 = ngal_split[:, b, i, 0]
                    half2 = ngal_split[:, b, i, 1]

                    # Convert to overdensity.  I thought about
                    # using half the mean from the full map to reduce
                    # noise, but thought that might add covariance
                    # to the two maps, and this shouldn't be that noisy
                    mu1 = np.average(half1, weights=mask[pixels])
                    mu2 = np.average(half2, weights=mask[pixels])
                    # This will produce some mangled sentinel values
                    # but they will be masked out
                    rho1 = (half1 - mu1) / mu1
                    rho2 = (half2 - mu2) / mu2

                    # Write both overdensity and count maps
                    # for each bin for each split
                    outfile.write_map(f"split_{i}/rho1_{b}", 
                        pixels, rho1, metadata)
                    outfile.write_map(f"split_{i}/rho2_{b}", 
                        pixels, rho2, metadata)
                    # counts
                    outfile.write_map(f"split_{i}/ngal1_{b}", 
                        pixels, half1, metadata)
                    outfile.write_map(f"split_{i}/ngal2_{b}", 
                        pixels, half2, metadata)
                    



    def read_inputs(self):

        with self.open_input('mask', wrapper=True) as f:
            mask = f.read_map('mask')
            # pixelization etc
            map_info = f.read_map_info('mask')

        with self.open_input('lens_maps', wrapper=True) as f:
            nbin_lens = f.file['maps'].attrs['nbin_lens']
            ngal_maps = [f.read_map(f'ngal_{b}') for b in range(nbin_lens)]

        with self.open_input('shear_tomography_catalog', wrapper=True) as f:
            nbin_source = f.file['tomography'].attrs['nbin_source']


        return nbin_source, nbin_lens, ngal_maps, mask, map_info


