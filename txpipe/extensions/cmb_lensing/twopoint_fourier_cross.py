from ...base_stage import PipelineStage
from ...data_types import MapsFile, SACCFile, TextFile, FiducialCosmology, PNGFile, QPNOfZFile
import numpy as np


class TXCMBLensingCrossMonteCarloCorrection(PipelineStage):
    """
    Compute the Monte Carlo normalization correction for the
    CMB lensing x galaxy density cross-correlation.

    Implements equation C.1 in https://arxiv.org/pdf/2407.04607 :

        (MC correction)_L =
            sum_ell W_Lell sum_i sum_m { M^kappa * kappa^i }_{lm} { M^g * kappa^i }*_{lm}
          / sum_ell W_Lell sum_i sum_m { M^kappa * kappa_hat^i }_{lm} { M^g * kappa^i }*_{lm}

    where:
      - kappa^i   = input CMB lensing convergence for simulation i
                    (from e.g. Planck FFP10 sky_klm_{i}.fits alm files)
      - kappa_hat^i = reconstructed CMB lensing convergence for simulation i
                    (from the CMB lensing reconstruction pipeline applied to sim i)
      - M^kappa   = CMB lensing mask
      - M^g       = galaxy survey mask
      - W_Lell    = NaMaster bandpower window function

    The numerator uses the input (true) kappa, the denominator uses the
    reconstructed kappa. Their ratio corrects for the bias introduced by
    the reconstruction pipeline (mode-couplings from masking and
    anisotropic filtering that MASTER does not forward-model).

    The output is a two-column text file (L_eff, MC_correction_L) to be
    read by TXTwoPointFourierCMBLensingCrossDensity.
    """

    name = "TXCMBLensingCrossMonteCarloCorrection"
    inputs = [
        ("cmb_lensing_map", MapsFile),
        ("mask", MapsFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [("cmb_cross_montecarlo_correction", TextFile)]

    config_options = {
        "cmb_redshift": 1100.0,
        "nside": 512,
        "nsim": 300,
        "mask_threshold": 0.0,
        "bandpower_width": 30,
        "lmax": 0,                    # 0 = use 3*nside - 1
        # Paths to simulation files.
        # input_kappa_alm_template : path with {i} placeholder for sim index
        #   e.g. "/path/to/COM_Lensing-SimMap-inputs/sky_klm_{i:04d}.fits"
        # recon_kappa_map_template : path with {i} placeholder
        #   e.g. "/path/to/recon_sims/kappa_recon_{i:04d}.fits"
        # sim_index_start : first simulation index (default 0)
        "input_phi_alm_template": str,
        "recon_kappa_alm_template": str,
        "sim_index_start": 0,
        "ffp10_offset": 200, # the FFP10 simulations correspond to the reconstruction with simidx+200
    }

    def run(self):
        import pymaster as nmt
        import healpy as hp

        nside = self.config["nside"]
        nsim  = self.config["nsim"]
        lmax  = self.config["lmax"] or (3 * nside - 1)
        i0    = self.config["sim_index_start"]

        print(f"TXCMBLensingCrossMonteCarloCorrection")
        print(f"  nside={nside}, nsim={nsim}, lmax={lmax}")

        # ---- 1. Load masks ----
        kappa_mask, galaxy_mask = self.load_masks()
        print(f"  Loaded masks")

        # ---- 2. Build NaMaster ell binning ----
        ell_bins  = self.make_ell_bins(nside, lmax)
        ell_eff   = ell_bins.get_effective_ells()
        n_ell     = len(ell_eff)
        print(f"  {n_ell} ell bins, ell_eff: [{ell_eff[0]:.0f}, {ell_eff[-1]:.0f}]")

        # ---- 3. Build the galaxy field (fixed across all sims) ----
        # The galaxy field is a unit map within the galaxy mask.
        # Its mask M^g is used in both numerator and denominator.
        galaxy_map = np.ones(hp.nside2npix(nside))
        galaxy_map[galaxy_mask == 0] = 0.0
        g_field = nmt.NmtField(galaxy_mask, [galaxy_map], n_iter=0)
        print(f"  Built galaxy NaMaster field")

        # ---- 4. Monte Carlo loop ----
        # Accumulate numerator and denominator sums over simulations.
        # For each sim i we need:
        #   numerator   += C_ell( M^kappa * kappa^i,     M^g * kappa^i )   [input x input]
        #   denominator += C_ell( M^kappa * kappa_hat^i, M^g * kappa^i )   [recon x input]
        #
        # NaMaster's compute_coupled_cell + decouple_cell computes the
        # windowed sum sum_ell W_Lell sum_m {A}_{lm} {B}*_{lm} in one call,
        # which is exactly the numerator/denominator structure of eq. C.1.

        num_sum = np.zeros(n_ell)
        den_sum = np.zeros(n_ell)

        for idx in range(nsim):
            i = i0 + idx
            if idx % 10 == 0:
                print(f"  Processing simulation {idx+1}/{nsim}  (index {i})")

            # Load input kappa^i and reconstructed kappa_hat^i
            kappa_input = self.load_input_kappa_map(self.config["ffp10_offset"] + i, nside, lmax)
            kappa_recon = self.load_recon_kappa_map(i, nside, lmax)

            # Pre-apply masks — the {AB}_{lm} notation in eq. C.1 means
            # the spherical harmonic transform of the masked map
            kappa_input_cmb = kappa_mask  * kappa_input   # M^kappa * kappa^i
            kappa_input_gal = galaxy_mask * kappa_input   # M^g     * kappa^i
            kappa_recon_cmb = kappa_mask  * kappa_recon   # M^kappa * kappa_hat^i

            # Numerator:   C_ell( M^kappa * kappa^i,     M^g * kappa^i )
            num_cl = self.compute_binned_cl(
                kappa_input_cmb, kappa_input_gal, ell_bins, lmax
            )
            num_sum += num_cl

            # Denominator: C_ell( M^kappa * kappa_hat^i, M^g * kappa^i )
            den_cl = self.compute_binned_cl(
                kappa_recon_cmb, kappa_input_gal, ell_bins, lmax
            )
            den_sum += den_cl

        # ---- 5. Compute the MC correction ----
        # MC_correction_L = numerator_L / denominator_L
        # Guard against division by zero
        mc_correction = np.where(
            np.abs(den_sum) > 0,
            num_sum / den_sum,
            1.0,
        )

        print(f"  MC correction range: [{mc_correction.min():.4f}, {mc_correction.max():.4f}]")

        # ---- 6. Write output ----
        output_path = self.get_output("cmb_cross_montecarlo_correction")
        header = (
            "Monte Carlo normalization correction (MC correction)_L\n"
            "Equation C.1 of https://arxiv.org/pdf/2407.04607\n"
            "numerator   = C_L( M^kappa * kappa^i,     M^g * kappa^i )  averaged over sims\n"
            "denominator = C_L( M^kappa * kappa_hat^i, M^g * kappa^i )  averaged over sims\n"
            f"nsim={nsim}, nside={nside}, lmax={lmax}\n"
            "Columns: L_eff   MC_correction"
        )
        np.savetxt(
            output_path,
            np.column_stack([ell_eff, mc_correction]),
            header=header,
        )
        print(f"  Written: {output_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_masks(self):
        """Load the CMB lensing mask and the galaxy survey mask."""
        import healpy as hp
        import pymaster as nmt

        with self.open_input("cmb_lensing_map", wrapper=True) as f:
            kappa_mask = f.read_map("kappa_mask")
        kappa_mask[kappa_mask == hp.UNSEEN] = 0.0
        # Rotate from Galactic to Celestial coordinates
        rot = hp.Rotator(coord=["G", "C"])
        kappa_mask = rot.rotate_map_pixel(kappa_mask)

        # Apodize. The size is the scale in degrees.
        # The type means the definition listed here:
        # https://namaster.readthedocs.io/en/latest/api/pymaster.utils.html#pymaster.utils.mask_apodization
        kappa_mask = nmt.mask_apodization(kappa_mask, aposize=0.2, apotype="C1")

        # Convert to a binary mask
        kappa_mask[kappa_mask > 0.5] = 1
        kappa_mask[kappa_mask <= 0.5] = 0

        with self.open_input("mask", wrapper=True) as f:
            galaxy_mask = f.read_mask(thresh=self.config["mask_threshold"])
        galaxy_mask[galaxy_mask == hp.UNSEEN] = 0.0

        return kappa_mask, galaxy_mask

    def make_ell_bins(self, nside, lmax):
        """Build a uniform NaMaster ell binning."""
        import pymaster as nmt

        bpw   = self.config["bandpower_width"]
        ells  = np.arange(0, lmax + 1, dtype=int)
        # Build edges: bins of width bpw starting from ell=2
        ell_ini = np.arange(2, lmax, bpw, dtype=int)
        ell_end = np.minimum(ell_ini + bpw, lmax + 1)
        b = nmt.NmtBin.from_edges(ell_ini, ell_end)
        return b

    def load_recon_kappa_map(self, sim_index, nside, lmax_out):
        """
        Load the lensing convergence kappa^i from an alm FITS file.
        Both for the input kappa^i and the reconstructed kappa_hat^i.

        Planck stores kappa_LM = ½ L(L+1) phi_LM as alm coefficients.
        We convert to a map at the requested nside.
        """
        import healpy as hp

        template = self.config["recon_kappa_alm_template"]
        
        path     = template.format(i=sim_index)

        # Read in the CMB lensing alms
        alms, lmax_in = hp.read_alm(path, return_mmax=True)

        # correct for the monopole
        alms[0] = 0 + 0j
        fl = np.ones(lmax_in+1)
        fl[lmax_out+1:] = 0
        alms = hp.almxfl(alms, fl, lmax_in)

        # rotate alms from Galactic to Celestial coordinates, since the Planck lensing sims are in Galactic 
        # but our pipeline is in Celestial. 
  
        rot = hp.Rotator(coord=["G", "C"]) 
        alms = rot.rotate_alm(alms)

        # Convert alm → map at nside
        kappa_map = hp.alm2map(alms, nside=nside, verbose=False)

        return kappa_map

    def load_input_kappa_map(self, sim_index, nside, lmax_out):
        """
        Load the input lensing potential phi^i from an alm FITS file,
        convert to kappa^i, and return a map at the requested nside.

        Planck FFP10 simulations store phi_LM as alm coefficients.
        We convert to kappa_LM = ½ L(L+1) phi_LM and then to a map.
        """
        import healpy as hp

        template = self.config["input_phi_alm_template"]
        
        path     = template.format(i=sim_index)

        # Read in the CMB lensing alms
        phi_alm, lmax_in = hp.read_alm(path, hdu=4, return_mmax=True)

        # Convert phi_LM → kappa_LM = ½ L(L+1) phi_LM
        ll = np.arange(lmax_in + 1)
        fl = ll * (ll + 1) / 2
        fl[lmax_out+1:] = 0
        kappa_alm = hp.almxfl(phi_alm, fl, lmax_in)

        # rotate alms from Galactic to Celestial coordinates, since the Planck lensing sims are in Galactic 
        # but our pipeline is in Celestial

        rot = hp.Rotator(coord=["G", "C"]) 
        kappa_alm = rot.rotate_alm(kappa_alm)

        # Convert alm → map at nside
        kappa_map = hp.alm2map(kappa_alm, nside=nside, verbose=False)

        return kappa_map

    def compute_binned_cl(self, map_a_masked, map_b_masked, ell_bins, lmax):
            """
            Compute the pseudo-C_ell between two masked maps using anafast,
            then bin to bandpowers using the NaMaster window function W_{Lell}.

            This implements the windowed sum:
                sum_ell W_{Lell} sum_m {A}_{lm} {B}*_{lm}

            which appears in both the numerator and denominator of eq. C.1.

            Parameters
            ----------
            map_a_masked : np.ndarray
                Map A with its mask already applied (M^kappa * kappa or M^kappa * kappa_hat).
            map_b_masked : np.ndarray
                Map B with its mask already applied (M^g * kappa).
            ell_bins : nmt.NmtBin
                NaMaster bandpower binning object — provides the window W_{Lell}.
            lmax : int
                Maximum multipole for anafast.

            Returns
            -------
            cl_binned : np.ndarray, shape (n_ell,)
                Pseudo-C_ell binned to bandpowers via W_{Lell}.
            """
            import healpy as hp

            # Compute pseudo-C_ell: sum_m {A}_{lm} {B}*_{lm}
            # anafast returns C_ell from ell=0 to lmax
            pcl = hp.anafast(map_a_masked, map_b_masked, lmax=lmax)

            # Bin using the NaMaster window function W_{Lell}
            # ell_bins.bin_cell expects shape (1, lmax+1) and returns (1, n_bands)
            ell = np.arange(len(pcl))
            cl_binned = ell_bins.bin_cell(pcl[np.newaxis, :])[0]

            return cl_binned


class TXTwoPointFourierCMBLensingCrossDensity(PipelineStage):
    """Compute the cross-correlation maps between CMB lensing and galaxy density."""

    name = "TXTwoPointFourierCMBLensingCrossDensity"

    inputs = [
        ("cmb_lensing_map", MapsFile),
        ("cmb_lensing_beam", TextFile),
        ("density_maps", MapsFile),
        ("density_masks", MapsFile),
        ("cmb_cross_montecarlo_correction", TextFile),
        ("lens_photoz_stack", QPNOfZFile),
    ]
    outputs = [
        ("twopoint_data_fourier_cmb_cross_density", SACCFile),
        ("twopoint_data_fourier_cmb_cross_density_plot", PNGFile),
    ]
    config_options = {
        "mask_threshold": 0.0,
        "bandpower_width": 30,
        "nside": 512,
    }

    def run(self):
        import pymaster as nmt
        import healpy as hp
        import scipy.interpolate
        import matplotlib.pyplot as plt
        import sacc
        from ...utils.nmt_utils import choose_ell_bins

        # make the field object for the CMB kappa map
        kappa_cmb, kappa_cmb_mask = self.load_kappa_maps()
        kappa_field = nmt.NmtField(kappa_cmb_mask, [kappa_cmb], n_iter=0)

        # Choose ell binning
        ell_bins = choose_ell_bins(**self.config)
        ell_eff = ell_bins.get_effective_ells()

        # make the field objects for the density maps
        d_maps, d_masks = self.load_density_maps()

        # load the Monte Carlo correction from equation equation 23 in https://arxiv.org/pdf/2407.04607
        transfer = self.load_monte_carlo_correction(ell_eff)

        # Compute the power spectrum for each density map x CMB kappa
        spectra = []
        workspaces = []
        windows = []
        d_fields = []
        for d_map, d_mask in zip(d_maps, d_masks):

            # Make a field from the loaded map and mask and compute the cross-spectrum
            # with the CMB kappa field.
            d_field = nmt.NmtField(d_mask, [d_map], n_iter=0)
            c_ell, workspace, window = self.compute_spectrum(kappa_field, d_field, ell_bins)

            # Scale by the transfer function
            c_ell /= transfer

            # Store all the results we have collected
            d_fields.append(d_field)
            spectra.append(c_ell)
            workspaces.append(workspace)
            windows.append(window)

        # Compute the covariance of all the fields we have loaded.
        covmat = self.compute_covariance(d_fields, kappa_field, workspaces)

        # make a plot for testing
        self.plot_spectra(spectra, covmat, ell_eff)

        # use the workspace to compute the covariance
        # write to SACC file
        self.save_spectra(spectra, ell_eff, covmat, windows)

    def compute_covariance(self, d_fields, kappa_field, workspaces):
        nlens = len(d_fields)
        cov_blocks = []

        # For now this module just does density x CMB kappa cross-spectra,
        # so there are no auto-spectra and the covariance is just between
        # different density bins and the cmb, so there are always nlens rows
        # and columns in the blocked covariance matrix.
        # We assemble the lower triangle here and then fill in the upper triangle
        # by symmetry afterwards.
        for i in range(nlens):
            cov_blocks_i = []
            field1 = kappa_field
            field2 = d_fields[i]
            for j in range(i + 1):
                field3 = kappa_field
                field4 = d_fields[j]
                fields = [field1, field2, field3, field4]
                print(f"Computing covariance block {i},{j}")
                block = self.get_covariance_block(fields, workspaces[i], workspaces[j])
                cov_blocks_i.append(block)

            cov_blocks.append(cov_blocks_i)

        # assemble the full covariance matrix
        for i in range(nlens):
            for j in range(i + 1, nlens):
                cov_blocks[i].append(cov_blocks[j][i])

        covmat = np.block(cov_blocks)

        return covmat

    def save_spectra(self, spectra, ell_eff, covmat, windows):
        """
        Save the computed spectra to a SACC file.
        """
        import sacc

        filename = self.get_output("twopoint_data_fourier_cmb_cross_density")

        # Use a standard data type spec for galaxy density x CMB convergence
        data_type = sacc.standard_types.cmbGalaxy_convergenceDensity_cl

        # This is the object we will be saving data to
        s = sacc.Sacc()

        # This is true for now because we are not doing the CMB x CMB auto-spectrum
        nbin_lens = len(spectra)
        with self.open_input("lens_photoz_stack", wrapper=True) as f:
            for i in range(nbin_lens):
                z, Nz = f.get_bin_n_of_z(i)
                s.add_tracer("NZ", f"lens_{i}", z, Nz)

        # Create the CMB kappa tracer, which stores the effective beam.
        # In the Quaia example this is just a unit beam again, following
        # the example Quaia notebook.
        with self.open_input("cmb_lensing_beam") as f:
            ell_beam, beam = np.loadtxt(f, unpack=True)
        ellmax = 3 * self.config["nside"] - 1
        lim = ell_beam <= ellmax
        ell_beam = ell_beam[lim]
        beam = beam[lim]

        # Create a CMB-type tracer with this beam
        s.add_tracer("Map", "cmb", spin=0, ell=ell_beam, beam=beam)

        # The bandpower windows from NaMaster seem to be defined up to the standard
        # 3 * Nside - 1 maximum multipole.
        ell_window = np.arange(ellmax + 1)

        for i, spectrum in enumerate(spectra):
            # Create a bandpower window for the current spectrum. The transpose
            # is needed because sacc expects shape (n_ell, n_band), not (n_band, n_ell)
            win = sacc.windows.BandpowerWindow(ell_window, windows[i].T)

            # save this data point. "cmb" refers to the CMB kappa tracer
            # name we created above, and similar for the lensing tracer
            s.add_ell_cl(data_type, "cmb", f"lens_{i}", ell_eff, spectrum, window=win)

        s.add_covariance(covmat)

        # Store provenance information in the sacc metadata
        provenance = self.gather_provenance()
        provenance.update(SACCFile.generate_provenance())

        # This is currently copied from the twopoint_fourier stage.
        # TODO: refactor to avoid duplication.
        for key, value in provenance.items():
            if isinstance(value, str) and "\n" in value:
                values = value.split("\n")
                for i, v in enumerate(values):
                    s.metadata[f"provenance/{key}_{i}"] = v
            else:
                s.metadata[f"provenance/{key}"] = value

        # Save as a FITS file
        s.save_fits(filename, overwrite=True)

    def get_covariance_block(self, fields, workspace12, workspace32):
        """
        Get the covariance block between
        C_ell[field[0], field[1]] and C_ell[field[2], field[3]]
        using NaMaster's Gaussian Covariance code.
        """
        import pymaster as nmt

        masks = [f.get_mask() for f in fields]
        # Compute all pseudo-Cls.
        # See the note here:
        # https://namaster.readthedocs.io/en/latest/api/pymaster.covariance.html#pymaster.covariance.gaussian_covariance
        cls02 = nmt.compute_coupled_cell(fields[0], fields[2]) / np.mean(masks[0] * masks[2])
        cls03 = nmt.compute_coupled_cell(fields[0], fields[3]) / np.mean(masks[0] * masks[3])
        cls12 = nmt.compute_coupled_cell(fields[1], fields[2]) / np.mean(masks[1] * masks[2])
        cls13 = nmt.compute_coupled_cell(fields[1], fields[3]) / np.mean(masks[1] * masks[3])

        cw = nmt.NmtCovarianceWorkspace.from_fields(fields[0], fields[1], fields[2], fields[3])
        spin0 = 0
        cv = nmt.gaussian_covariance(
            cw,
            spin0,
            spin0,
            spin0,
            spin0,
            cls02,
            cls03,
            cls12,
            cls13,
            workspace12,
            wb=workspace32,
        )
        return cv

    def plot_spectra(self, spectra, covmat, ell_eff):
        """
        Make a quick plot of the C_ell for testing.
        """
        nlens = len(spectra)
        sigmas = np.sqrt(np.diag(covmat))
        sigmas = np.array_split(sigmas, nlens)
        n = len(spectra)
        figsize = (5, 5 * n)
        with self.open_output(
            "twopoint_data_fourier_cmb_cross_density_plot", wrapper=True, figsize=figsize
        ) as f:
            for i, (c_ell, sigma) in enumerate(zip(spectra, sigmas)):
                ax = f.file.add_subplot(n, 1, i + 1)

                ax.errorbar(ell_eff, c_ell, yerr=sigma, fmt=".", label=f"Density map {i} X CMB Kappa")
                ax.set_xscale("log")
                ax.set_yscale("log")
                if i == n - 1:
                    ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(r"$C_\ell$")
                if i == 0:
                    ax.legend()

    def compute_spectrum(self, field1, field2, ell_bins):
        """Compute the cross-spectrum between two fields."""
        import pymaster as nmt

        # Compute the Pseudo-C_ell
        pcl = nmt.compute_coupled_cell(field1, field2)

        # Decouple the power spectrum to turn it into an estimate of the true power spectrum
        # These are all cross-correlations, so there is no noise bias
        workspace = nmt.NmtWorkspace.from_fields(field1, field2, ell_bins)
        c_ell = workspace.decouple_cell(pcl)[0]
        windows = workspace.get_bandpower_windows()
        return c_ell, workspace, windows.squeeze()

    def load_kappa_maps(self):
        """Load the CMB kappa and mask maps."""
        import healpy

        # Load the CMB lensing map and mask
        with self.open_input("cmb_lensing_map", wrapper=True) as f:
            kappa_cmb = f.read_map("kappa_cmb")
            kappa_cmb_mask = f.read_map("kappa_mask")

        # Set unseen pixels in the mask to zero to correctly downweight them
        unseen_to_zero(kappa_cmb_mask)

        return kappa_cmb, kappa_cmb_mask

    def load_density_maps(self):
        import healpy as hp

        # Read the over-density maps
        with self.open_input("density_maps", wrapper=True) as f:
            nbin_lens = f.file["maps"].attrs["nbin_lens"]
            d_maps = [f.read_map(f"delta_{b}") for b in range(nbin_lens)]

        # Read the masks, one per density bin
        with self.open_input("density_masks", wrapper=True) as f:
            masks = [
                f.read_mask(f"mask_{b}", thresh=self.config["mask_threshold"]) for b in range(nbin_lens)
            ]

        # Check that the density maps have no unseen values under the mask.
        # If they do then this leaks into the power spectrum and the results are wildly wrong.
        for m, d in zip(masks, d_maps):
            assert not np.any(d[m > 0] == hp.UNSEEN), "Density map has unseen values under the mask"

        # In the masks any unseen pixels should be set to zero weight
        for m in masks:
            unseen_to_zero(m)

        return d_maps, masks

    def load_monte_carlo_correction(self, ell):
        from scipy.interpolate import interp1d

        with self.open_input("cmb_cross_montecarlo_correction") as f:
            ell_transfer, transfer = np.loadtxt(f, unpack=True)

            # Take the first or last value if we go out of range
            fill_value = transfer[0], transfer[-1]
            transfer_interp = interp1d(ell_transfer, transfer, bounds_error=False, fill_value=fill_value)
        return transfer_interp(ell)


def unseen_to_zero(m):
    import healpy as hp

    m[m == hp.UNSEEN] = 0.0
    return m
