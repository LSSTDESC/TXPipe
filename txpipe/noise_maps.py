from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, DiagnosticMaps, \
                        NoiseMaps, HDFFile
import numpy as np

class TXNoiseMaps(PipelineStage):
    """
    Generate a suite of random noise maps by randomly
    rotating individual galaxy measurements.

    """
    name='TXNoiseMaps'

    
    inputs = [
        ('shear_catalog', HDFFile),
        ('tomography_catalog', TomographyCatalog),
        # We get the pixelization info from the diagnostic maps
        ('diagnostic_maps', DiagnosticMaps),
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
        bin_cols = ['source_bin', 'lens_bin']

        # Make the iterators
        chunk_rows = self.config['chunk_rows']
        shear_it = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        bin_it = self.iterate_hdf('tomography_catalog','tomography', bin_cols, chunk_rows)
        bin_it = (d[2] for d in bin_it)

        npix = pixel_scheme.npix

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
        ngal_split = np.zeros((npix, nbin_lens, clustering_realizations, 2))
        # TODO: Clustering weights go here

        # Loop through the data
        for (s, e, shear_data), bin_data in zip(shear_it, bin_it):
            print(f"Rank {self.rank} processing rows {s} - {e}")
            source_bin = bin_data['source_bin']
            lens_bin = bin_data['lens_bin']
            ra = shear_data['ra']
            dec = shear_data['dec']
            pixels = pixel_scheme.ang2pix(ra, dec)

            n = e - s

            w = shear_data['weight']
            g1 = shear_data['mcal_g1'] * w
            g2 = shear_data['mcal_g2'] * w

            # randomly select a half for each object
            split = np.random.binomial(1, 0.5, (n, clustering_realizations))

            # random rotations of the g1, g2 values
            phi = np.random.uniform(0, 2*np.pi, (n, lensing_realizations))
            c = np.cos(phi)
            s = np.sin(phi)
            g1r =  c * g1[:, np.newaxis] + s * g2[:, np.newaxis]
            g2r = -s * g1[:, np.newaxis] + c * g2[:, np.newaxis]

            for i in range(n):
                sb = source_bin[i]
                lb = lens_bin[i]
                pix = pixels[i]
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
            from mpi4py.MPI import DOUBLE, SUM, IN_PLACE
            if self.comm.Get_rank() == 0:
                self.comm.Reduce(IN_PLACE, G1)
                self.comm.Reduce(IN_PLACE, G2)
                self.comm.Reduce(IN_PLACE, GW)
                self.comm.Reduce(IN_PLACE, ngal_split)
            else:
                self.comm.Reduce(G1, None)
                self.comm.Reduce(G2, None)
                self.comm.Reduce(GW, None)
                self.comm.Reduce(ngal_split, None)
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

            for b in range(nbin_source):
                pixels = np.where(GW[:,b]>0)[0]
                for i in range(lensing_realizations):

                    g1 = G1[pixels, b, i] / GW[pixels, b]
                    g2 = G2[pixels, b, i] / GW[pixels, b]

                    outfile.write_map(f"rotation_{i}/g1_{b}", 
                        pixels, g1, metadata)

                    outfile.write_map(f"rotation_{i}/g2_{b}", 
                        pixels, g2, metadata)

            for b in range(nbin_lens):
                pixels = np.where(mask>0)[0]
                for i in range(clustering_realizations):
                    # We have computed the first half already,
                    # and we have the total map from an earlier stage
                    half1 = ngal_split[:, b, i, 0]
                    half2 = ngal_split[:, b, i, 1]

                    # Convert to overdensity.  I thought about
                    # using half the mean from the full map to reduce
                    # noise, but thought that might add covariance
                    # to the two maps, and this shouldn't be that noisy
                    mu1 = np.average(half1, weights=mask)
                    mu2 = np.average(half2, weights=mask)
                    # This will produce some mangled sentinel values
                    # but they will be masked out
                    rho1 = (half1 - mu1) / mu1
                    rho2 = (half2 - mu2) / mu2

                    # Write both overdensity and count maps
                    # for each bin for each split
                    outfile.write_map(f"split_{i}/rho1_{b}", 
                        pixels, rho1[pixels], metadata)
                    outfile.write_map(f"split_{i}/rho2_{b}", 
                        pixels, rho2[pixels], metadata)
                    # counts
                    outfile.write_map(f"split_{i}/ngal1_{b}", 
                        pixels, half1[pixels], metadata)
                    outfile.write_map(f"split_{i}/ngal2_{b}", 
                        pixels, half2[pixels], metadata)
                    



    def read_inputs(self):

        map_file = self.open_input('diagnostic_maps', wrapper=True)

        # Get the bin counts
        info = map_file.file['maps'].attrs
        nbin_source = info['nbin_source']
        nbin_lens = info['nbin_lens']

        # get pixelization info from the usual maps.
        map_info = map_file.read_map_info('mask')

        # Get the ngal values, so we can subtract the random
        # half selection from it to get the other half
        ngal_maps = [map_file.read_map(f'ngal_{b}') for b in range(nbin_lens)]

        # and the mask, which we will use to decide which pixels to keep
        mask = map_file.read_mask()

        map_file.close()

        return nbin_source, nbin_lens, ngal_maps, mask, map_info


