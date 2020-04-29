from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, DiagnosticMaps, \
                        LensingNoiseMaps, ClusteringNoiseMaps, HDFFile
import numpy as np

class TXLensingNoiseMaps(PipelineStage):
    """
    Generate a suite of random noise maps by randomly
    rotating individual galaxy measurements.

    """
    name='TXLensingNoiseMaps'

    
    inputs = [
        ('shear_catalog', HDFFile),
        ('tomography_catalog', TomographyCatalog),
        # We get the pixelization info from the diagnostic maps
        ('diagnostic_maps', DiagnosticMaps),
    ]

    outputs = [
        ('lensing_noise_maps', LensingNoiseMaps),
    ]

    config_options = {
        'chunk_rows': 100000,
        'n_realization': 30,
    }        

    def run(self):
        from .utils import choose_pixelization

        # get the number of bins.
        bins, map_info = self.read_metadata()
        nbin = len(bins)
        pixel_scheme = choose_pixelization(**map_info)
        n_rotations = self.config['n_realization']

        # The columns we will need
        shear_cols = ['ra', 'dec', 'weight', 'mcal_g1', 'mcal_g2']
        bin_cols = ['source_bin']

        # Make the iterators
        chunk_rows = self.config['chunk_rows']
        shear_it = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        bin_it = self.iterate_hdf('tomography_catalog','tomography', bin_cols, chunk_rows)
        bin_it = (d[2] for d in bin_it)

        npix = pixel_scheme.npix

        if self.rank == 0:
            nGB = (npix * nbin * n_rotations * 24) / 1024.**3
            print(f"Allocating maps of size {nGB:.2f} GB") 

        G1 = np.zeros((npix, nbin, n_rotations))
        G2 = np.zeros((npix, nbin, n_rotations))
        W = np.zeros((npix, nbin))

        # Loop through the data
        for (s, e, shear_data), bin_data in zip(shear_it, bin_it):
            print(f"Rank {self.rank} processing rows {s} - {e}")
            source_bin = bin_data['source_bin']
            ra = shear_data['ra']
            dec = shear_data['dec']
            pixels = pixel_scheme.ang2pix(ra, dec)

            n = e - s

            w = shear_data['weight']
            g1 = shear_data['mcal_g1'] * w
            g2 = shear_data['mcal_g2'] * w

            phi = np.random.uniform(0, 2*np.pi, (n, n_rotations))
            c = np.cos(phi)
            s = np.sin(phi)
            g1r =  c * g1[:, np.newaxis] + s * g2[:, np.newaxis]
            g2r = -s * g1[:, np.newaxis] + c * g2[:, np.newaxis]

            for i in range(n):
                sb = source_bin[i]
                if sb >= 0:
                    pix = pixels[i]
                    G1[pix, sb, :] += g1r[i] 
                    G2[pix, sb, :] += g2r[i]
                    W[pix, sb] += w[i]

        # Sum everything at root
        if self.comm is not None:
            from mpi4py.MPI import DOUBLE, SUM, IN_PLACE
            if self.comm.Get_rank() == 0:
                self.comm.Reduce(IN_PLACE, G1)
                self.comm.Reduce(IN_PLACE, G2)
                self.comm.Reduce(IN_PLACE, W)
            else:
                self.comm.Reduce(G1, None)
                self.comm.Reduce(G2, None)
                self.comm.Reduce(W, None)


        if self.rank==0:
            print("Saving maps")
            outfile = self.open_output('lensing_noise_maps', wrapper=True)

            # The top section has the metadata in
            group = outfile.file.create_group("maps")
            group.attrs['nbin_source'] = nbin
            group.attrs['n_realization'] = n_rotations

            metadata = {**self.config, **map_info}

            for b in range(nbin):
                pixels = np.where(W[:,b]>0)[0]
                for i in range(n_rotations):

                    g1 = G1[pixels, b, i] / W[pixels, b]
                    g2 = G2[pixels, b, i] / W[pixels, b]

                    outfile.write_map(f"realization_{i}/g1_{b}", 
                        pixels, g1, metadata)

                    outfile.write_map(f"realization_{i}/g2_{b}", 
                        pixels, g2, metadata)



    def read_metadata(self):
        # get pixelization info from the usual maps.
        map_file = self.open_input('diagnostic_maps', wrapper=True)
        map_info = map_file.read_map_info('lensing_weight_0')
        map_file.close()

        # Get the bin count from tomography.
        tomo_file = self.open_input('tomography_catalog', wrapper=False)
        info = tomo_file['tomography'].attrs
        nbin = info['nbin_source']
        tomo_file.close()

        bins = list(range(nbin))

        return bins, map_info


class TXClusteringNoiseMaps(PipelineStage):
    name='TXClusteringNoiseMaps'
    
    inputs = [
        ('diagnostic_maps', DiagnosticMaps),
    ]

    outputs = [
        ('clustering_noise_maps', ClusteringNoiseMaps),
    ]

    config_options = {
        'n_realization': 30,
    }        

    def run(self):
        # Input and output file.
        map_file = self.open_input('diagnostic_maps', wrapper=True)
        out_file = self.open_output('clustering_noise_maps', wrapper=True)
        group = out_file.file.create_group('maps')
        n_realization = self.config['n_realization']

        # Map info - nside, etc.
        map_info = map_file.read_map_info('mask')
        # The mask - as of now this is just binary, but
        # will be gradually improved
        mask = map_file.read_map('mask')

        # Count of bins.  We just do lensing in this
        # one, so ignore the nbin_source
        _, nbin = map_file.get_nbins()

        
        group.attrs['nbin_source'] = nbin
        group.attrs['n_realization'] = n_realization

        # To be saved in the output
        metadata = {**self.config, **map_info}

        # make a randomizer objects which prepares
        # the probabilities per pixel
        randomizer = MapRandomizer(mask)
        pixel = randomizer.pixel

        for b in range(nbin):
            print(f"Simulating random clustering for bin {b}")
            ngal = map_file.read_map(f'ngal_{b}')

            # The mask can be smaller than the ngal map
            # if we have set some regions as masked, or if
            # there is a count threshold, for example.  We
            # don't want to move galaxies from outside the mask
            # region into it.
            ngal[mask <= 0] = 0

            ntot = int(ngal[ngal>0].sum())

            # Loop realizations
            for i in range(n_realization):
                # Generate a random map with this ngal
                random_ngal, random_delta = randomizer(ntot)

                # Save the maps
                out_file.write_map(f'realization_{i}/ngal_{b}', pixel, random_ngal, metadata)
                out_file.write_map(f'realization_{i}/delta_{b}', pixel, random_delta, metadata)



        if self.rank == 0:
            print("NOTE: Using mask from diagnostic_maps.  Just uniform for now.")



class MapRandomizer:
    def __init__(self, mask):
        self.mask = mask
        self.hit = mask > 0
        self.mask_hit = mask[self.hit]
        self.nhit = self.mask_hit.size
        self.pixel = np.arange(mask.size)[self.hit]
        self.mask_pix = np.arange(self.nhit, dtype=int)
        self.pix_prob = self.mask_hit / self.mask_hit.sum()

    def __call__(self, ngal):
        galpix = np.random.choice(self.mask_pix, size=ngal, p=self.pix_prob)
        count_map = np.bincount(galpix, minlength=self.nhit)
        mu = count_map.mean()
        delta_map = (count_map - mu) / mu
        return count_map, delta_map
