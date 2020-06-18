import numpy as np

from .base_stage import PipelineStage
from .data_types import MapsFile


class TXSimpleMask(PipelineStage):
    name = "TXSimpleMask"
    # make a mask from the auxiliary maps
    inputs = [('aux_maps', MapsFile)]
    outputs = [('mask', MapsFile)]
    config_options = {
        'depth_cut' : 23.5,
        'bright_object_max': 10.0,
    }        

    def run(self):
        with self.open_input('aux_maps', wrapper=True) as f:
            metadata = dict(f.file['maps'].attrs)
            bright_obj = f.read_map('bright_objects/count')
            depth = f.read_map('depth/depth')

        mask = (
            (depth > self.config['depth_cut']) 
            & 
            (bright_obj < self.config['bright_object_max'])
        )
        f_sky = (mask.sum() * 1.0) / mask.size
        area = f_sky * 41252.96125
        print(f"f_sky = {f_sky}")
        print(f"area = {area:.1f} sq deg")
        metadata['area'] = area
        metadata['f_sky'] = f_sky

        # Pull out unmasked pixels and their values
        pix = np.where(mask)[0]
        mask = mask[pix].astype(float)

        mask[np.isnan(mask)] = 0.0
        mask[mask < 0] = 0

        with self.open_output('mask', wrapper=True) as f:

            f.file.create_group('maps')
            f.write_map('mask', pix, mask, metadata)
