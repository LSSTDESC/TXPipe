from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, DiagnosticMaps, NoiseMaps, HDFFile
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
        ('lensing_noise_maps', NoiseMaps),
    ]

    config_options = {
        'chunk_rows': 100000,
        'n_rotations': 30,
        'sparse': True,
    }        

    def run(self):
        from .mapping import ShearNoiseMapper
        from .utils import choose_pixelization
        from healsparse import HealSparseMap

        # get the number of bins.
        bins, map_info = self.read_metadata()
        pixel_scheme = choose_pixelization(**map_info)
        n_rotations = self.config['n_rotations']

        # The columns we will need
        shear_cols = ['ra', 'dec', 'weight', 'mcal_g1', 'mcal_g2']
        bin_cols = ['source_bin']

        # Make the iterators
        chunk_rows = self.config['chunk_rows']
        shear_it = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        bin_it = self.iterate_hdf('tomography_catalog','tomography', bin_cols, chunk_rows)
        bin_it = (d[2] for d in bin_it)

        npix = pixel_scheme.npix

        G1 = np.zeros((npix, nbin_source, n_rotations))
        G2 = np.zeros((npix, nbin_source, n_rotations))
        W = np.zeros((npix, nbin_source))

        # Loop through the data
        for (s, e, shear_data), bin_data in zip(shear_it, bin_it):
            source_bin = bin_data['source_bin']

            n = s - e
            w = shear_data['weight']
            g1 = shear_data['mcal_g1'] * w
            g2 = shear_data['mcal_g2'] * w

            phi = np.random.uniform(0, 2*np.pi, (ngal, n_rotations))
            c = np.cos(phi)
            s = np.sin(phi)
            g1r =  c * g1[:, np.newaxis] + s * g2[:, np.newaxis]
            g2r = -s * g1[:, np.newaxis] + c * g2[:, np.newaxis]

            for i in range(ngal):
                if source_bin >= 0:
                    pix = pixels[i]
                    G1[pix, source_bin, :] += g1r[i] 
                    G2[pix, source_bin, :] += g2r[i]
                    W[pix, source_bin] += w[i]

        # Sum everything at root
        if self.comm is not None:
            from mpi4py.MPI import DOUBLE, SUM
            if self.comm.Get_rank() == 0:
                self.comm.Reduce(MPI.IN_PLACE, G1)
                self.comm.Reduce(MPI.IN_PLACE, G2)
                self.comm.Reduce(MPI.IN_PLACE, W)
            else:
                self.comm.Reduce(G1, None)
                self.comm.Reduce(G2, None)
                self.comm.Reduce(W, None)


        if self.rank==0:
            print("Saving maps")
            outfile = self.open_output('lensing_noise_maps', wrapper=True)

            # The top section has the metadata in
            group = outfile.file.create_group("maps")
            group.attrs['nbin_source'] = len(bins)
            group.attrs['n_realizations'] = n_rotations

            metadata = {**self.config, **map_info}

            for b in range(nbin_source):
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
