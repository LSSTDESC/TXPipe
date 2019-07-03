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

        float_params =[
            "mjd-obs",
            "filter",
            "testtype",
            "imgtype",
            "ratel",
            "dectel",
            "rotangle",
            "bgmean",
            "bgvar",
            "magzero_rms",
            "colorterm1",
            "colorterm2",
            "colorterm3",
            "fluxmag0",
            "fluxmag0err",
            "exptime",
            "darktime",
            
        ]

        int_params = [
            "runnum",
            "obsid",
            "magzero_nobj",
            "ap_corr_map_id",
            "psf_id",
            "skywcs_id",
            "psf_id",
            "skywcs_id",
        ]

        str_params  = [
            "date-avg",
            "timesys",
            "rottype",
        ]


        # Spaces for output columns
        n = len(refs)

        data =  {p:np.zeros(n) for p in float_params}
        data += {p:np.zeros(n, type=int) for p in int_params}
        data += {list() for p in str_params}

        num_params = float_params + int_params


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
            for p in num_params:
                data[p][i] = hdr[p.upper()]
            for p in str_params:
                data[p].append(hdr[p.upper()])

        # Save output
        f = self.open_output('field_centers')
        g = f.create_group('field_centers')
        for name, col in data.items():
            g.create_dataset(name, data=col)
         f.close()
