from ...base_stage import PipelineStage
from ...data_types import MapsFile, SACCFile, TextFile, FiducialCosmology
import numpy as np

class TXCMBLensingCrossMonteCarloCorrection(PipelineStage):
    """Compute equation 23 in https://arxiv.org/pdf/2407.04607"""
    inputs = [
        ("cmb_lensing_map", MapsFile),
        ("mask", MapsFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [
        ("cmb_cross_montecarlo_correction", TextFile)
    ]

    config_options = {
        "cmb_redshift": 1100.0,
        "nside": 512,
        "nsim": 1000,
        "mask_threshold": 0.0,
    }

    def run(self):
        raise ValueError("This stage is not yet implemented. It needs CMB kappa reconstruction realizations.")
        ell, c_ell = self.compute_fiducial_cmb_kappa()
        cmb_mask, galaxy_mask = self.load_masks()


    def load_masks(self):
        with self.open_input("cmb_lensing_map") as f:
            kappa_cmb_mask = f.read_map("kappa_mask")
        
        with self.open_input("mask", wrapper=True) as f:
            galaxy_mask = f.read_mask(thresh=self.config["mask_threshold"])

        return kappa_cmb_mask, galaxy_mask

    def simulate_kappa(self):
        # for each pair of maps in the reconstruction
        pass


class TXTwoPointFourierCMBLensingCrossDensity(PipelineStage):
    """Compute the cross-correlation maps between CMB lensing and galaxy density.
    
    """

    inputs = [
        ("cmb_lensing_map", MapsFile),
        ("density_maps", MapsFile),
        ("mask", MapsFile),
        ("cmb_cross_montecarlo_correction", TextFile)
    ]
    outputs = [
        ("twopoint_data_fourier_cmb_cross", SACCFile),
        ("fourier_cmb_cross_plot", SACCFile)
        
        ]
    config_options = {
        "mask_threshold": 0.0,
        "bandpower_width": 30,
    }

    def run(self):
        import pymaster as nmt
        import healpy as hp
        import scipy.interpolate
        import matplotlib.pyplot as plt

        # make the field object for the CMB kappa map
        kappa_cmb, kappa_cmb_mask = self.load_kappa_maps()
        kappa_field = nmt.NmtField(kappa_cmb_mask, [kappa_cmb], n_iter=0)

        # Choose ell binning
        npix = len(kappa_cmb_mask)
        nside = hp.npix2nside(npix)
        ell_bins = self.choose_ell_bins(nside)
        ell_eff = ell_bins.get_effective_ells()

        # make the field objects for the density maps
        d_maps, d_mask = self.load_density_maps()

        # load the Monte Carlo correction from equation equation 23 in https://arxiv.org/pdf/2407.04607
        transfer = self.load_monte_carlo_correction(ell_eff)

        # Compute the power spectrum for each density map x CMB kappa
        spectra = []
        for d_map in d_maps:
            d_field = nmt.NmtField(d_mask, [d_map], n_iter=0)            
            c_ell, workspace, window = self.compute_spectrum(kappa_field, d_field, ell_bins)
            c_ell /= transfer
            spectra.append(c_ell)

        # make a plot for testing
        self.plot_spectra(spectra, ell_eff)

        # use the workspace to compute the covariance
        # write to SACC file

    def plot_spectra(self, spectra, ell_eff):
        n = len(spectra)
        figsize = (5, 5*n)
        with self.open_output("fourier_cmb_cross_plot", wrapper=True, figsize=figsize) as f:
            for i, c_ell in enumerate(spectra):
                ax = f.file.add_subplot(n, 1, i+1)

                # make a plot for testing
                ax[i].loglog(ell_eff, c_ell, '.', label=f"Density map {i} X CMB Kappa")
                if i == n - 1:
                    ax[i].set_xlabel(r"$\ell$")
                ax[i].set_ylabel(r"$C_\ell$")
                if i == 0:
                    ax[i].legend()


    def choose_ell_bins(self, nside):
        """Choose the ell bins for the power spectrum computation."""
        import pymaster as nmt
        bandpower_width = self.config["bandpower_width"]
        ell_bins = nmt.NmtBin.from_nside_linear(nside, nlb=bandpower_width)
        return ell_bins

    def compute_spectrum(self, field1, field2, ell_bins):
        """Compute the cross-spectrum between two fields."""
        import pymaster as nmt

        # Compute the Pseudo-C_ell
        pcl = nmt.compute_coupled_cell(field1, field2)

        # Decouple the power spectrum to turn it into an estimate of the true power spectrum
        # These are cross-correlations, so there is no noise bias
        workspace = nmt.NmtWorkspace()
        workspace.compute_coupling_matrix(field1, field2, ell_bins)
        c_ell = workspace.decouple_cell(pcl)

        windows = workspace.get_bandpower_windows()
        return c_ell, workspace, windows.squeeze()


    def load_kappa_maps(self):
        """Load the CMB kappa and mask maps."""
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

    def load_monte_carlo_correction(self, ell):
        from scipy.interpolate import interp1d

        with self.open_input("cmb_cross_montecarlo_correction") as f:
            ell_transfer, transfer = np.loadtxt(f, unpack=True)
            t0, t1 = transfer[0], transfer[-1]
            # Take the first or last value if we go out of range
            transfer_interp = interp1d(ell_transfer, transfer, bounds_error=False, fill_value=[t0, t1])
        return transfer_interp(ell)