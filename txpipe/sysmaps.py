from pipette import PipelineStage
from descformats.tx import MetacalCatalog, DiagnosticMaps, YamlFile


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
        ('shear_catalog', MetacalCatalog),
        ('config', YamlFile),
    ]
    
    # We generate a single HDF file in this stage
    # containing all the maps
    outputs = [
        ('diagnostic_maps', DiagnosticMaps),
    ]
    
    # Configuration information for this stage
    config_options = {
        'nside':None,   # The Healpix resolution parameter for the generated maps
        'snr_threshold':None,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        'snr_delta':1.0,  # The range threshold +/- delta is used for finding objects at the boundary
        'chunk_rows':100000,  # The number of rows to read in each chunk of data at a time
        'sparse':None,   # Whether to generate sparse maps - faster and less memory for small sky areas
    }


    def run(self):
        from .depth import dr1_depth
        import numpy as np

        # Read input configuration informatiomn
        config = self.read_config()

        # Set up the iterator to run through the FITS file.
        # Iterators lazily load the data chunk by chunk as we iterate through the file.
        # We don't need to use the start and end points in this case, as 
        # we're not making a new catalog.
        cat_cols = ['ra', 'dec', 'mcal_s2n_r', 'mcal_mag']
        data_iterator = (data for start,end,data in self.iterate_fits('shear_catalog', 1, cat_cols, config['chunk_rows']))

        # Calculate the depth map, map of the counts used in computing the depth map, and map of the depth variance
        pixel, count, depth, depth_var = dr1_depth(data_iterator, 
            config['nside'], config['snr_threshold'], config['snr_delta'], sparse=config['sparse'], 
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
        """
        subgroup = group.create_group(name)
        subgroup.attrs.update(metadata)
        subgroup.create_dataset("pixel", data=pixel)
        subgroup.create_dataset("value", data=value)






if __name__ == '__main__':
    PipelineStage.main()
