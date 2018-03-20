from pipette import PipelineStage
from descformats.tx import MetacalCatalog, DiagnosticMaps, YamlFile


class TXDiagnosticMaps(PipelineStage):
    name='TXDiagnosticMaps'
    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('config', YamlFile),
    ]
    outputs = [
        ('diagnostic_maps', DiagnosticMaps),
    ]
    config_options = {'nside':None, 'snr_threshold':None, 'chunk_rows':100000, 'sparse':None}


    def run(self):
        from .depth import dr1_depth

        # Set up the calculator
        config = self.read_config()
        nside = config['nside']
        snr_threshold = config['snr_threshold']
        sparse = config['sparse']

        # Set up the iterator to run through the FITS file.
        # We don't need the start and end points in this case
        cat_cols = ['ra', 'dec', 'mcal_s2n_r', 'mcal_mag']
        chunk_rows = config['chunk_rows']
        data_iterator = (data for start,end,data in self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows))

        # Calculate the depth from 
        count, depth, depth_var = dr1_depth(data_iterator, nside, snr_threshold, sparse=sparse, comm=self.comm)


        if self.rank==0:
            outfile = self.open_output('diagnostic_maps')
            maps_group = outfile.create_group("maps")
            depth_maps_group = maps_group.create_group("depth")
            if sparse:
                w = count.nonzero()
                pixel = w[0]
                depth_maps_group.create_dataset("pixel", data=pixel)
                depth_maps_group.create_dataset("depth", data=depth[w].toarray()[0,:])
                depth_maps_group.create_dataset("count", data=count[w].toarray()[0,:])
                depth_maps_group.create_dataset("depth_var", data=depth_var[w].toarray()[0,:])
            else:
                depth_maps_group.create_dataset("depth", data=depth)
                depth_maps_group.create_dataset("count", data=count)
                depth_maps_group.create_dataset("depth_var", data=depth_var)






if __name__ == '__main__':
    PipelineStage.main()
