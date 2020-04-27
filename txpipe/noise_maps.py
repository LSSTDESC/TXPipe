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
        from .mapping import Mapper
        from .utils import choose_pixelization

        # get the number of bins.
        bins, map_info = self.read_metadata()
        pixel_scheme = choose_pixelization(**map_info)
        n_rotations = self.config['n_rotations']

        # use the same mapper object as the other mapping stage to build the maps
        mappers = [
            # No lens bins here, only source
            Mapper(pixel_scheme, [], bins, sparse=self.config['sparse'])
            for i in range(n_rotations)
        ]

        # The columns we will need
        shear_cols = ['ra', 'dec', 'weight', 'mcal_g1', 'mcal_g2']
        bin_cols = ['source_bin']

        # Make the iterators
        chunk_rows = self.config['chunk_rows']
        shear_it = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        bin_it = self.iterate_hdf('tomography_catalog','tomography', bin_cols, chunk_rows)
        bin_it = (d[2] for d in bin_it)

        # Loop through the data
        for (s, e, shear_data), bin_data in zip(shear_it, bin_it):
            print(f"Process {self.rank} random rotating rows {s:,}-{e:,}")

            n = len(shear_data['ra'])

            # make a random rotation for each galaxy
            for i in range(n_rotations):
                g1 = shear_data['mcal_g1']
                g2 = shear_data['mcal_g2']

                # Generate the rotation values
                phi = np.random.uniform(0, 2*np.pi, n)
                s = np.sin(phi)
                c = np.cos(phi)

                shear_data['g1'] =  g1 * c + g2 * s
                shear_data['g2'] = -g1 * s + g2 * c

                # The "None" is for the m values, which are not currently
                # used but included as a placeholder for when/if we want
                # to start mapping the response values
                mappers[i].add_data(shear_data, bin_data, None)

        if self.rank==0:
            print("Saving maps")
            outfile = self.open_output('lensing_noise_maps', wrapper=True)

            # The top section has the metadata in
            group = outfile.file.create_group("maps")
            group.attrs['nbin_source'] = len(bins)
            group.attrs['n_realizations'] = n_rotations

            metadata = {**self.config, **map_info}


        for i, mapper in enumerate(mappers):
            if self.rank == 0:
                print(f"Collating and saving map rotation {i}")
            # First collate the map info from the other processors,
            # if we are running in parallel.
            # We don't need weight maps, because they are the same as
            # the original one.  We also don't need the density map.
            map_pix, _, g1, g2, _, _, _ = mapper.finalize(self.comm)

            if self.rank == 0:
                # Save the collated map for this rotation
                for b in bins:
                    outfile.write_map(f"realization_{i}/g1_{b}", 
                                  map_pix, g1[b], metadata)
                    outfile.write_map(f"realization_{i}/g2_{b}", 
                                  map_pix, g2[b], metadata)



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
