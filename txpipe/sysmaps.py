from ceci import PipelineStage
from .data_types import HDFFile, DiagnosticMaps, YamlFile
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
        'nside':0,   # The Healpix resolution parameter for the generated maps. Only req'd if using healpix
        'snr_threshold':float,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        'snr_delta':1.0,  # The range threshold +/- delta is used for finding objects at the boundary
        'chunk_rows':100000,  # The number of rows to read in each chunk of data at a time
        'sparse':True,   # Whether to generate sparse maps - faster and less memory for small sky areas,
        'ra_cent':np.nan,  # These parameters are only required if pixelization==tan
        'dec_cent':np.nan,  
        'npix_x':-1, 
        'npix_y':-1, 
        'pixel_size':np.nan, # Pixel size of pixelization scheme
        'depth_band' : 'i',
    }


    def run(self):
        """
        Run the analysis for this stage.

         - choose the pixelization scheme for the map
         - loop through chunks of the photometry catalog (in paralllel if enabled)
         - build up the map gradually
         - the master process saves the map
        """
        from .depth import dr1_depth
        from .utils import choose_pixelization
        import numpy as np

        # Read input configuration informatiomn
        config = self.config


        # Select a pixelization scheme based in configuration keys.
        # Looks at "pixelization as the main thing"
        pixel_scheme = choose_pixelization(**config)
        config.update(pixel_scheme.metadata)

        band = config['depth_band']
        cat_cols = ['ra', 'dec', f'snr_{band}', f'mag_{band}_lsst']


        # This bit may be confusing!  To avoid loading all the data
        # at once we use an iterator, which loads the data chunk by chunk.
        # We will pass this iterator to the depth code, but we also need to rename
        # two columns in it so that we know which band to use for the depth.
        # In this case piggyback on an existing code that iterates through the file (iterate_hdf)
        # but also add additional columns, called mag and snr.
        # This is all probably typical Joe overkill.
        def iterate():
            for _,_,data in self.iterate_hdf('photometry_catalog', 'photometry', cat_cols, 
                                              config['chunk_rows']):
                data['mag'] = data[f'mag_{band}_lsst']
                data['snr'] = data[f'snr_{band}']
                yield data

        # Now make an instance of this iterator
        data_iterator = iterate()


        # Calculate the depth map, map of the counts used in computing the depth map
        # and map of the depth variance.
        pixel, count, depth, depth_var = dr1_depth(data_iterator,
            pixel_scheme,
            config['snr_threshold'],
            config['snr_delta'],
            sparse=config['sparse'],
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

            # I'm expecting this will one day call off to a 10,000 line
            # library or something.
            mask, npix = self.compute_mask(count)
            self.save_map(group, "mask", pixel, mask, config)

            area = pixel_scheme.pixel_area(degrees=True) * npix
            group.attrs['area'] = area
            group.attrs['area_unit'] = 'sq deg'


    def compute_mask(self, depth_count):
        mask = np.zeros_like(depth_count)
        hit = depth_count > 0
        mask[hit] = 1.0
        count = hit.sum()
        return mask, hit



    def save_map(self, group, name, pixel, value, metadata):
        """
        Save an output map to an HDF5 subgroup.

        The pixel numbering and the metadata are also saved.

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
