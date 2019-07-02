from .base_stage import PipelineStage
from .data_types import HDFFile
import numpy as np

class TXButlerFieldCenters(PipelineStage):
    """
    """
    name='TXButlerFieldCenters'

    inputs = [
    ]
    outputs = [
        ('field_centers', HDFFile),
    ]
    config = {'dc2_name': str}

    def run(self):
        # find butler by name using this tool.
        # change later to general repo path.
        from desc_dc2_dm_data import get_butler
        butler = get_butler(self.config['dc2_name'])

        # central detector in whole focal plane, just as example
        exposure_refs = butler.subset('calexp', raftName='R22', detectorName='S11')

        # Spaces for output columns
        n = len(refs)
        ra = np.zeros(n)
        dec = np.zeros(n)
        rot = np.zeros(n)
        obs_id = np.zeros(n, dtype=int)
        run_num = np.zeros(n, dtype=int)
        exp_id = np.zeros(n, dtype=int)

        # Loop through. Much faster to access the file
        # directly rather than through ref.get, which seems
        # to load in the full image 
        for i,ref in enumerate(refs):
            # Progress update
            if i%20==0:
                print(i)
            # Open the file for this reference
            filename = ref.getUri()
            f = fits.open(fn)
            hdr = f[0].header
            # columns that we want, pulled out of FITS headers.
            ra[i] = hdr["RATEL"]
            dec[i] = hdr["DECTEL"]
            rot[i] = hdr['ROTANGLE']
            obs_id[i] = hdr['OBSID']
            run_num[i] = hdr['RUNNUM']
            exp_id[i] = hdr['EXPID']

        # Save output
        f = self.open_output('field_centers')
        g = f.create_group('field_centers')
        g.create_dataset('ra', data=ra)
        g.create_dataset('dec', data=dec)
        g.create_dataset('rot', data=rot)
        g.create_dataset('run_num', data=run_num)
        g.create_dataset('obs_id', data=obs_id)
        g.create_dataset('exp_id', data=exp_id)
        f.close()
