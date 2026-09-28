import numpy as np
import os

from .base_stage import PipelineStage
from .data_types import (
    MapsFile,
    HDFFile,
    QPNOfZFile,
)
from ceci.config import StageParameter
from .utils.nmt_utils import choose_ell_bins


class TXSystematicsTwoPointFourier(PipelineStage):
    """
    Cross-correlate galaxy density maps with systematics maps in Fourier space.

    Produces two SACC files: one where the galaxy density fields have the
    systematics modes deprojected (via NaMaster templates), and one without
    deprojection. Both files include a Gaussian covariance matrix computed with
    NaMaster. This is useful as a null-test: after deprojection, cross-spectra
    should be consistent with zero for any systematics map that was used as a
    template.

    Inputs
    ------
    lens_photoz_stack : QPNOfZFile
        N(z) distributions for the lens tomographic bins.
    tracer_metadata : HDFFile
        Galaxy counts per lens bin (used for shot-noise in the covariance).
    density_maps : MapsFile
        Galaxy overdensity maps delta_{b} for each lens bin b.
    mask : MapsFile
        Survey footprint mask.

    Outputs
    -------
    summary_statistics_systematics_fourier_deproj : SACCFile
        Cross-spectra computed with systematics deprojection applied to the
        galaxy density fields, plus Gaussian covariance.
    summary_statistics_systematics_fourier_no_deproj : SACCFile
        Same cross-spectra without deprojection, plus Gaussian covariance.
    """

    name = "TXSystematicsTwoPointFourier"

    inputs = [
        ("lens_photoz_stack", QPNOfZFile),
        ("tracer_metadata", HDFFile),
        ("density_maps", MapsFile),
        ("mask", MapsFile),
    ]

    outputs = [
        ("summary_statistics_systematics_fourier_deproj", HDFFile),
        ("summary_statistics_systematics_fourier_no_deproj", HDFFile),
    ]

    config_options = {
        "systmaps_clustering_dir": StageParameter(str, "", msg="Systematics maps: a directory, glob pattern or path prefix (HEALPix or HealSparse files)"),
        "systmaps_healsparse_reduction": StageParameter(
            str, "mean",
            msg="Reduction method when degrading HealSparse maps (mean/median/std/max/min/sum/prod)"
        ),
        "mask_threshold": StageParameter(float, 0.0, msg="Mask pixel threshold"),
        "ell_min": StageParameter(int, 100, msg="Minimum ell"),
        "ell_max": StageParameter(int, 1500, msg="Maximum ell"),
        "n_ell": StageParameter(int, 20, msg="Number of ell bins"),
        "ell_spacing": StageParameter(str, "log", msg="Ell spacing: log or linear"),
        "cache_dir": StageParameter(
            str, "./cache/systematics_twopoint_fourier", msg="Directory for cached NaMaster workspaces"
        ),
    }

    def run(self):
        import pymaster as nmt
        import healpy as hp

        # ------------------------------------------------------------------ #
        # 1.  Load mask and density maps
        # ------------------------------------------------------------------ #
        with self.open_input("mask", wrapper=True) as f:
            info = f.read_map_info("mask")
            mask = f.read_mask(thresh=self.config["mask_threshold"])

        f_sky = info["f_sky"]
        nside = hp.get_nside(mask)
        print(f"Mask nside={nside}, f_sky={f_sky:.4f}")

        with self.open_input("density_maps", wrapper=True) as f:
            nbin_lens = f.file["maps"].attrs["nbin_lens"]
            d_maps = [f.read_map(f"delta_{b}") for b in range(nbin_lens)]
        print(f"Loaded {nbin_lens} density maps")

        # ------------------------------------------------------------------ #
        # 2.  Load systematics maps
        # ------------------------------------------------------------------ #
        s_maps, syst_names = self._read_systematics_maps(mask, nside)
        n_syst = len(s_maps)
        if n_syst == 0:
            raise RuntimeError(
                "No systematics maps found in systmaps_clustering_dir="
                f"'{self.config['systmaps_clustering_dir']}'. "
                "This stage requires at least one systematics map."
            )
        print(f"Loaded {n_syst} systematics maps")

        # ------------------------------------------------------------------ #
        # 3.  Read lens counts for shot noise
        # ------------------------------------------------------------------ #
        with self.open_input("tracer_metadata") as f:
            lens_counts = f["tracers/lens_counts"][:]
            area_deg2 = f["tracers"].attrs["area"]

        area_sr = area_deg2 * (np.pi / 180.0) ** 2
        n_bar = lens_counts / area_sr  # mean galaxy density per steradian

        # ------------------------------------------------------------------ #
        # 4.  NaMaster fields
        # ------------------------------------------------------------------ #
        lmax = self.config["ell_max"] - 1
        s_maps_nmt = np.array(s_maps).reshape([n_syst, 1, hp.nside2npix(nside)])

        density_fields_no_deproj = [
            nmt.NmtField(mask, [d], n_iter=0, lmax=lmax) for d in d_maps
        ]
        density_fields_deproj = [
            nmt.NmtField(mask, [d], templates=s_maps_nmt, n_iter=0, lmax=lmax) for d in d_maps
        ]
        syst_fields = [
            nmt.NmtField(mask, [s_maps[x]], n_iter=0, lmax=lmax) for x in range(n_syst)
        ]
        print("Created NaMaster fields")

        # ------------------------------------------------------------------ #
        # 5.  Ell bins
        # ------------------------------------------------------------------ #
        ell_bins = choose_ell_bins(**self.config)
        ell_eff = ell_bins.get_effective_ells()
        n_ell = len(ell_eff)
        ell_window = np.arange(lmax + 1)
        print(f"Using {n_ell} ell bins: {ell_eff[0]:.0f} – {ell_eff[-1]:.0f}")

        # ------------------------------------------------------------------ #
        # 6.  Shared NmtWorkspace  (all density and syst maps share the same mask)
        # ------------------------------------------------------------------ #
        cache_dir = self.config["cache_dir"]
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        workspace_path = os.path.join(cache_dir, "w_density_syst.fits") if cache_dir else None

        w = nmt.NmtWorkspace()
        if workspace_path and os.path.exists(workspace_path):
            print(f"Loading workspace from cache: {workspace_path}")
            w.read_from(workspace_path)
        else:
            print("Computing NaMaster workspace …")
            w.compute_coupling_matrix(density_fields_no_deproj[0], syst_fields[0], ell_bins)
            if workspace_path:
                w.write_to(workspace_path)

        bandpower_wins = w.get_bandpower_windows()  # shape (1, n_ell, 1, lmax+1)

        # ------------------------------------------------------------------ #
        # 7.  Compute all required power spectra
        #     We need three sets of pseudo-C_ells for the Gaussian covariance:
        #       dd[a,b]  = density_a × density_b  (includes shot noise on diag)
        #       ds[a,x]  = density_a × syst_x     (our main output)
        #       ss[x,y]  = syst_x   × syst_y
        # ------------------------------------------------------------------ #
        mean_mask2 = float(np.mean(mask ** 2))

        def _pseudo_cl(fa, fb):
            """Coupled C_ell divided by mean(mask^2) — input to gaussian_covariance."""
            return nmt.compute_coupled_cell(fa, fb) / mean_mask2  # shape (1, lmax+1)

        def _decoupled_cl(fa, fb):
            """Decoupled, binned C_ell."""
            pcl = nmt.compute_coupled_cell(fa, fb)
            return w.decouple_cell(pcl)[0]  # shape (n_ell,)

        print("Computing density × density pseudo-C_ells …")
        cl_dd_pseudo = np.zeros((nbin_lens, nbin_lens, lmax + 1))
        for a in range(nbin_lens):
            for b in range(a, nbin_lens):
                cl = _pseudo_cl(density_fields_no_deproj[a], density_fields_no_deproj[b])[0]
                if a == b:
                    # Add shot noise: N_ell = 1/n_bar (flat in ell)
                    cl = cl + 1.0 / n_bar[a]
                cl_dd_pseudo[a, b] = cl
                cl_dd_pseudo[b, a] = cl

        print("Computing density × syst pseudo-C_ells …")
        # No-deproj and deproj versions both needed
        cl_ds_pseudo_no_deproj = np.zeros((nbin_lens, n_syst, lmax + 1))
        cl_ds_pseudo_deproj   = np.zeros((nbin_lens, n_syst, lmax + 1))
        cl_ds_decoupled_no_deproj = np.zeros((nbin_lens, n_syst, n_ell))
        cl_ds_decoupled_deproj   = np.zeros((nbin_lens, n_syst, n_ell))

        for a in range(nbin_lens):
            for x in range(n_syst):
                cl_ds_pseudo_no_deproj[a, x] = _pseudo_cl(
                    density_fields_no_deproj[a], syst_fields[x]
                )[0]
                cl_ds_pseudo_deproj[a, x] = _pseudo_cl(
                    density_fields_deproj[a], syst_fields[x]
                )[0]
                cl_ds_decoupled_no_deproj[a, x] = _decoupled_cl(
                    density_fields_no_deproj[a], syst_fields[x]
                )
                cl_ds_decoupled_deproj[a, x] = _decoupled_cl(
                    density_fields_deproj[a], syst_fields[x]
                )

        print("Computing syst × syst pseudo-C_ells …")
        cl_ss_pseudo = np.zeros((n_syst, n_syst, lmax + 1))
        for x in range(n_syst):
            for y in range(x, n_syst):
                cl = _pseudo_cl(syst_fields[x], syst_fields[y])[0]
                cl_ss_pseudo[x, y] = cl
                cl_ss_pseudo[y, x] = cl

        # ------------------------------------------------------------------ #
        # 8.  Gaussian covariance workspace  (one for all spin-0 quadruplets)
        # ------------------------------------------------------------------ #
        cov_workspace_path = (
            os.path.join(cache_dir, "cw_density_syst.fits") if cache_dir else None
        )
        cw = nmt.NmtCovarianceWorkspace()
        if cov_workspace_path and os.path.exists(cov_workspace_path):
            print(f"Loading covariance workspace from cache: {cov_workspace_path}")
            cw.read_from(cov_workspace_path)
        else:
            print("Computing NaMaster covariance workspace …")
            cw.compute_coupling_coefficients(
                density_fields_no_deproj[0], syst_fields[0],
                density_fields_no_deproj[0], syst_fields[0],
            )
            if cov_workspace_path:
                cw.write_to(cov_workspace_path)

        # ------------------------------------------------------------------ #
        # 9.  Build full covariance matrix for all (lens_a × syst_x) pairs
        # ------------------------------------------------------------------ #
        # Data vector ordering: outer loop over lens bins, inner over syst maps
        # index mapping: i_pair(a, x) = a * n_syst + x
        n_cross = nbin_lens * n_syst
        cov_no_deproj = np.zeros((n_cross * n_ell, n_cross * n_ell))
        cov_deproj   = np.zeros((n_cross * n_ell, n_cross * n_ell))

        print(f"Computing Gaussian covariance ({n_cross} cross-spectra × {n_ell} ell bins) …")
        for a in range(nbin_lens):
            for x in range(n_syst):
                ax = a * n_syst + x
                for b in range(nbin_lens):
                    for y in range(n_syst):
                        by = b * n_syst + y

                        # Only compute upper triangle; mirror below
                        if by < ax:
                            continue

                        # No-deproj covariance block
                        block_no_deproj = nmt.gaussian_covariance(
                            cw, 0, 0, 0, 0,
                            cl_dd_pseudo[a, b].reshape(1, -1),
                            cl_ds_pseudo_no_deproj[a, y].reshape(1, -1),
                            cl_ds_pseudo_no_deproj[b, x].reshape(1, -1),
                            cl_ss_pseudo[x, y].reshape(1, -1),
                            wa=w, wb=w,
                            coupled=False,
                        )  # shape (n_ell, n_ell)

                        # Deproj covariance block
                        block_deproj = nmt.gaussian_covariance(
                            cw, 0, 0, 0, 0,
                            cl_dd_pseudo[a, b].reshape(1, -1),
                            cl_ds_pseudo_deproj[a, y].reshape(1, -1),
                            cl_ds_pseudo_deproj[b, x].reshape(1, -1),
                            cl_ss_pseudo[x, y].reshape(1, -1),
                            wa=w, wb=w,
                            coupled=False,
                        )

                        row = slice(ax * n_ell, (ax + 1) * n_ell)
                        col = slice(by * n_ell, (by + 1) * n_ell)
                        cov_no_deproj[row, col] = block_no_deproj
                        cov_deproj[row, col]   = block_deproj
                        if ax != by:
                            cov_no_deproj[col, row] = block_no_deproj.T
                            cov_deproj[col, row]   = block_deproj.T

        print("Covariance computation done")

        # ------------------------------------------------------------------ #
        # 10. Load lens N(z) for metadata
        # ------------------------------------------------------------------ #
        lens_nz = {}
        with self.open_input("lens_photoz_stack", wrapper=True) as f:
            for b in range(nbin_lens):
                z, Nz = f.get_bin_n_of_z(b)
                lens_nz[b] = (z, Nz)

        # ------------------------------------------------------------------ #
        # 11. Save results as plain HDF5
        # ------------------------------------------------------------------ #
        # sacc's HDF5/FITS writers cannot handle O(1000) data points reliably
        # (FITS column limit, HDF5 compact-storage object-header overflow).
        # Write directly with h5py: everything needed for plotting and chi-sq.

        import h5py

        def _save_hdf5(path, cl_ds, cov_matrix, label):
            with h5py.File(path, "w") as f:
                f.attrs["label"] = label
                f.attrs["n_lens_bins"] = nbin_lens
                f.attrs["n_syst_maps"] = n_syst
                f.attrs["systmaps_clustering_dir"] = self.config["systmaps_clustering_dir"]

                f.create_dataset("ell_eff", data=ell_eff)
                # cl shape: (n_lens_bins, n_syst_maps, n_ell)
                f.create_dataset("cl", data=cl_ds)
                # cov shape: (n_lens_bins*n_syst_maps*n_ell, same)
                f.create_dataset("cov", data=cov_matrix)

                dt = h5py.special_dtype(vlen=str)
                ds = f.create_dataset("syst_names", (n_syst,), dtype=dt)
                for i, name in enumerate(syst_names):
                    ds[i] = name

                nz_grp = f.create_group("lens_nz")
                for b, (z, Nz) in lens_nz.items():
                    nz_grp.create_dataset(f"z_{b}", data=z)
                    nz_grp.create_dataset(f"nz_{b}", data=Nz)

        out_no_deproj = self.get_output("summary_statistics_systematics_fourier_no_deproj")
        out_deproj   = self.get_output("summary_statistics_systematics_fourier_deproj")

        _save_hdf5(out_no_deproj, cl_ds_decoupled_no_deproj, cov_no_deproj, "no_deprojection")
        _save_hdf5(out_deproj,   cl_ds_decoupled_deproj,   cov_deproj,   "deprojection")
        print(f"Saved no-deproj HDF5 → {out_no_deproj}")
        print(f"Saved deproj HDF5    → {out_deproj}")

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _read_systematics_maps(self, mask, nside):
        """Read the survey-property maps as mean-subtracted NaMaster templates.

        Uses the shared reader in txpipe.utils.systematics_maps; `nside` must
        match the mask's. Returns a list of full-sky RING arrays and their
        source filenames.
        """
        import healpy as hp
        from .utils.systematics_maps import read_systematics_maps

        assert hp.get_nside(mask) == nside
        if not self.config["systmaps_clustering_dir"]:
            print("systmaps_clustering_dir not set; no systematics maps will be loaded.")
            return [], []
        return read_systematics_maps(
            self.config["systmaps_clustering_dir"],
            mask,
            reduction=self.config["systmaps_healsparse_reduction"],
        )
