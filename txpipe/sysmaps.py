from ceci import PipelineStage
from descformats.tx import HDFFile, DiagnosticMaps, YamlFile
import numpy as np

class TXDiagnosticMaps(PipelineStage):
    """
    For now, this Pipeline Stage computes a depth map using the DR1 method,
    which takes the mean magnitude of objects close to 5-sigma S/N.

    In the future we will add the calculation of other diagnostic maps
    like airmass for use in systematics tests and covariance mode projection.

    DM may in the future provide tools we can use in place of the methods
    used here, but not on the DC2 timescale.

    """
    name='TXDiagnosticMaps'

    # We currently take everything from the shear catalog.
    # In the long run this may become DM output
    inputs = [
        ('photometry_catalog', HDFFile),
    ]

    # We generate a single HDF file in this stage
    # containing all the maps
    outputs = [
        ('diagnostic_maps', DiagnosticMaps),
    ]

    # Configuration information for this stage
    config_options = {
        'pixelization': 'healpix', # The pixelization scheme to use, currently just healpix
        'nside':0,   # The Healpix resolution parameter for the generated maps
        'snr_threshold':float,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        'snr_delta':1.0,  # The range threshold +/- delta is used for finding objects at the boundary
        'chunk_rows':100000,  # The number of rows to read in each chunk of data at a time
        'sparse':True,   # Whether to generate sparse maps - faster and less memory for small sky areas,
        'ra_min':np.nan,  #
        'ra_max':np.nan,  # RA range
        'dec_min':np.nan, #
        'dec_max':np.nan, # DEC range
        'pixel_size':np.nan, # Pixel size of pixelization scheme
        'depth_band' : 'i',

    }


    def run(self):
        from .depth import dr1_depth
        from .utils import choose_pixelization
        import numpy as np

        # Read input configuration informatiomn
        config = self.config


        # Select a pixelization scheme based in configuration keys.
        # Looks at "pixelization as the main thing"
        pixel_scheme = choose_pixelization(**config)
        config.update(pixel_scheme.metadata)

        # Set up the iterator to run through the FITS file.
        # Iterators lazily load the data chunk by chunk as we iterate through the file.
        # We don't need to use the start and end points in this case, as
        # we're not making a new catalog.
        band = config['depth_band']
        cat_cols = ['ra', 'dec', f'snr_{band}', f'mag_{band}_lsst']
        def iterate():
            for start,end,data in self.iterate_hdf('photometry_catalog', 'photometry', cat_cols, 
                                                    config['chunk_rows']):
                data['mag'] = data[f'mag_{band}_lsst']
                data['snr'] = data[f'snr_{band}']
                yield data

        data_iterator = iterate()


        # Calculate the depth map, map of the counts used in computing the depth map, and map of the depth variance
        pixel, count, depth, depth_var = dr1_depth(data_iterator,
            pixel_scheme, config['snr_threshold'], config['snr_delta'], sparse=config['sparse'],
            comm=self.comm)
        
        # Only the root process saves the output
        if self.rank==0:
            # Open the HDF5 output file
            outfile = self.open_output('diagnostic_maps')
            # Use one global section for all the maps
            group = outfile.create_group("maps")
            # Save each of the maps in a separate subsection
            self.save_map(group, "depth", pixel, depth, config)
            self.save_map(group, "depth_count", pixel, count, config)
            self.save_map(group, "depth_var", pixel, depth_var, config)



    def save_map(self, group, name, pixel, value, metadata):
        """
        Save an output map to an HDF5 subgroup, including the pixel
        numbering and the metadata.

        Parameters
        ----------

        group: H5Group
            The h5py Group object in which to store maps
        name: str
            The name of this map, used as the name of a subgroup in the group where the data is stored.
        pixel: array
            Array of indices of observed pixels
        value: array
            Array of values of observed pixels
        metadata: mapping
            Dict or other mapping of metadata to store along with the map
        """
        subgroup = group.create_group(name)
        subgroup.attrs.update(metadata)
        subgroup.create_dataset("pixel", data=pixel)
        subgroup.create_dataset("value", data=value)






if __name__ == '__main__':
    PipelineStage.main()
