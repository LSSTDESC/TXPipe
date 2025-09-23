import numpy as np
from ...base_stage import PipelineStage
from ...data_types import MapsFile, TextFile


class TXIngestPlanckLensingMaps(PipelineStage):
    """Ingest Planck NPIPE CMB lensing maps.

    This stage reads in the Planck PR4 CMB lensing maps, and mask, and
    noise spectrum.

    Input files:
    - data/planck/PR4_klm_dat_p.fits: The Planck PR4 CMB lensing alms.
    - data/planck/mask.fits.gz: The Planck PR4 CMB lensing mask.
    - data/planck/PR4_klm_dat_p_noise.fits: The Planck PR4 CMB lensing noise spectrum.
    """

    name = "TXIngestPlanckLensingMaps"
    inputs = []
    outputs = [
        ("cmb_lensing_map", MapsFile)
    ]
    config_options = {
        "alm_file": "data/planck/PR4_klm_dat_p.fits",
        "mask_file": "data/planck/mask.fits.gz",
        "noise_file": "data/planck/PR4_klm_dat_p_noise.fits",
    }

    def run(self):
        import healpy

        # Read in the CMB lensing alms and mask
        alms = healpy.read_alm("data/planck/PR4_klm_dat_p.fits")
        mask = healpy.read_map("data/planck/mask.fits.gz")
        nside = healpy.get_nside(mask)
        alms[0] = 0
        kappa_map = healpy.alm2map(alms, nside=nside)

        kappa_pix = np.where(mask > 0)[0]
        kappa_val = kappa_map[kappa_pix]
        mask_val = mask[kappa_pix]

        kappa_noise_spectrum = np.loadtxt("data/planck/PR4_klm_dat_p_noise.fits")
        ell = np.arange(len(kappa_noise_spectrum))

        with self.open_output("cmb_lensing_map", wrapper=True) as f:
            f.write_map("kappa_cmb", kappa_pix, kappa_val)
            f.write_map("kappa_mask", kappa_pix, mask_val)

            # add a separate section for the noise
            g = f.file.create_group("noise_spectrum")
            g.create_dataset("noise_n_ell", data=kappa_noise_spectrum)
            g.create_dataset("ell", data=ell)
