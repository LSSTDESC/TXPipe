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
    config_options = {'dc2_name': str}

    def run(self):
        from astropy.io import fits

        # find butler by name using this tool.
        # change later to general repo path.
        from desc_dc2_dm_data import get_butler
        run = self.config['dc2_name']
        print(f"Getting butler for repo {run}")
        butler = get_butler(run)
        print("Butler loaded")

        # central detector in whole focal plane, just as example
        refs = butler.subset('calexp', raftName='R22', detectorName='S11')
        n = len(refs)
        print(f"Found {n} exposure centers.  Reading metadata.")

        # Spaces for output columns
        ra = np.zeros(n)
        dec = np.zeros(n)
        rot = np.zeros(n)
        obs_id = np.zeros(n, dtype=int)
        run_num = np.zeros(n, dtype=int)
        exp_id = np.zeros(n, dtype=int)
        mjd = np.zeros(n)

        # Loop through. Much faster to access the file
        # directly rather than through ref.get, which seems
        # to load in the full image 
        for i,ref in enumerate(refs):
            # Progress update
            if i%20==0:
                print(i)
            # Open the file for this reference
            filename = ref.getUri()
            f = fits.open(filename)
            hdr = f[0].header
            # columns that we want, pulled out of FITS headers.
            ra[i] = hdr["RATEL"]
            dec[i] = hdr["DECTEL"]
            rot[i] = hdr['ROTANGLE']
            obs_id[i] = hdr['OBSID']
            run_num[i] = hdr['RUNNUM']
            exp_id[i] = hdr['EXPID']
            mjd[i] = hdr['MJD-OBS']

        # Save output
        f = self.open_output('field_centers')
        g = f.create_group('field_centers')
        g.create_dataset('ra', data=ra)
        g.create_dataset('dec', data=dec)
        g.create_dataset('rot', data=rot)
        g.create_dataset('run_num', data=run_num)
        g.create_dataset('obs_id', data=obs_id)
        g.create_dataset('exp_id', data=exp_id)
        g.create_dataset('mjd_obs', data=mjd)
        f.close()
