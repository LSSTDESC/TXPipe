from ...twopoint_fourier import TXTwoPointFourier
from ...data_types import MapsFile, SACCFile, TextFile

class TXTwoPointFourierCMBLensingCross(TXTwoPointFourier):
    """Compute the cross-correlation maps between CMB lensing and galaxy shear.
    
    """

    inputs = TXTwoPointFourier.inputs + [
        ("cmb_lensing_map", MapsFile),
        ]
    outputs = [("twopoint_data_fourier_cmb_cross", SACCFile)]
    config_options = TXTwoPointFourier.config_options | {}


    def load_maps(self):
        import healpy

        with self.open_input("external_maps_file") as f:
            kappa_cmb = f.read_map("kappa_cmb")
            kappa_mask = f.read_map("kappa_mask")
            noise_n_ell = f.file["noise_spectrum/noise_n_ell"][:]
            noise_ell = f.file["noise_spectrum/ell"][:]

        pixel_scheme, maps, f_sky = super().load_maps()
        maps["cmb_kappa"] = kappa_cmb
        maps["cmb_kappa_mask"] = kappa_mask
        maps["cmb_kappa_noise_n_ell"] = noise_n_ell
        maps["cmb_kappa_noise_ell"] = noise_ell

        return pixel_scheme, maps, f_sky