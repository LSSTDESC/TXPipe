from ...base_stage import PipelineStage
from ...data_types import MapsFile, SACCFile
import numpy as np

class TXTwoPointFourierCMBLensingCrossDensity(PipelineStage):
    """Compute the cross-correlation maps between CMB lensing and galaxy density.
    
    """

    inputs = [
        ("cmb_lensing_map", MapsFile),
        ("density_maps", MapsFile),
        ("mask", MapsFile),
    ]
    outputs = [("twopoint_data_fourier_cmb_cross", SACCFile)]
    config_options = {
        "mask_threshold": 0.0,
        "bandpower_width": 30,
    }

    def run(self):
        import pymaster as nmt
        import healpy as hp

        # make the field object for the CMB kappa map
        kappa_cmb, kappa_cmb_mask = self.load_kappa_maps()
        kappa_field = nmt.NmtField(kappa_cmb_mask, [kappa_cmb], n_iter=0)

        # Choose ell binning
        npix = len(kappa_cmb_mask)
        nside = hp.npix2nside(npix)
        ell_bins = self.choose_ell_bins(nside)

        # make the field objects for the density maps
        d_maps, d_mask = self.load_density_maps()

        # Compute the power spectrum for each density map x CMB kappa
        for d_map in d_maps:
            d_field = nmt.NmtField(d_mask, [d_map], n_iter=0)            
            c_ell, n_ell, workspace, window = self.compute_spectrum(kappa_field, d_field, ell_bins)

    def choose_ell_bins(self, nside):
        import pymaster as nmt
        bandpower_width = self.config["bandpower_width"]
        ell_bins = nmt.NmtBin.from_nside_linear(nside, nlb=bandpower_width)
        return ell_bins

    def compute_spectrum(self, field1, field2, ell_bins):
        import pymaster as nmt
        pcl = nmt.compute_coupled_cell(field1, field2)
        workspace = nmt.NmtWorkspace()
        workspace.compute_coupling_matrix(field1, field2, ell_bins)
        c_ell = workspace.decouple_cell(pcl)
        n_ell = np.zeros_like(c_ell)
        windows = workspace.get_bandpower_windows()
        return c_ell, n_ell, workspace, windows.squeeze()


    def load_kappa_maps(self):
        import healpy

        with self.open_input("cmb_lensing_map") as f:
            kappa_cmb = f.read_map("kappa_cmb")
            kappa_cmb_mask = f.read_map("kappa_mask")
        
        return kappa_cmb, kappa_cmb_mask
    


    def load_density_maps(self):
        with self.open_input("density_maps", wrapper=True) as f:
            nbin_lens = f.file["maps"].attrs["nbin_lens"]
            d_maps = [f.read_map(f"delta_{b}") for b in range(nbin_lens)]

        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_mask(thresh=self.config["mask_threshold"])

        return d_maps, mask
