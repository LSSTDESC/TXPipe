from .twopoint import TXTwoPoint, TREECORR_CONFIG, SHEAR_POS
from .base_stage import PipelineStage
from .data_types import SACCFile, ShearCatalog, HDFFile, QPNOfZFile, FiducialCosmology, TextFile, PNGFile
import numpy as np
from ceci.config import StageParameter
import os


class TXDeltaSigma(TXTwoPoint):
    """Compute Delta-Sigma, the excess surface density around lenses.
    This version uses the dSigma code.
        """

    name = "TXDeltaSigma"

    inputs = [
        ("binned_shear_catalog", ShearCatalog),
        ("binned_lens_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        # ("tracer_metadata", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [("delta_sigma", SACCFile)]

    config_options = TREECORR_CONFIG | {
        # "source_bins": StageParameter(list, [-1], msg="List of source bins to use (-1 means all)"),
        # "lens_bins": StageParameter(list, [-1], msg="List of lens bins to use (-1 means all)"),
        # "var_method": StageParameter(str, "jackknife", msg="Method for computing variance (jackknife, sample, etc.)"),
        # "use_randoms": StageParameter(bool, True, msg="Whether to use random catalogs"),
        # "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode"),
        # "patch_dir": StageParameter(str, "./cache/delta-sigma-patches", msg="Directory for storing patch files"),
        # "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk when making patches"),
        # "share_patch_files": StageParameter(bool, False, msg="Whether to share patch files across processes"),
        # "metric": StageParameter(str, "Euclidean", msg="Distance metric to use (Euclidean, Arc, etc.)"),
        # "gaussian_sims_factor": StageParameter(
        #     list,
        #     default=[1.0],
        #     msg="Factor by which to decrease lens density to account for increased density contrast.",
        # ),
        # # "use_subsampled_randoms": StageParameter(bool, False, msg="Use subsampled randoms file for RR calculation"),
        # "delta_z": StageParameter(float, 0.001, msg="Z bin width for sigma_crit spline computation"),
    }

    def run(self):
        import dSigma
        


class TXDeltaSigmaPlots(PipelineStage):
    """Make plots of Delta Sigma results.
    
    """
    name = "TXDeltaSigmaPlots"
    inputs = [
        ("delta_sigma", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),
        
    ]
    outputs = [
        ("delta_sigma_plot", PNGFile),
        ("delta_sigma_r_plot", PNGFile),
]
    config_options = {}

    def run(self):
        import sacc
        import matplotlib.pyplot as plt

        sacc_data = sacc.Sacc.load_fits(self.get_input("delta_sigma"))

        # Plot in theta coordinates
        nbin_source = sacc_data.metadata['nbin_source']
        nbin_lens = sacc_data.metadata['nbin_lens']
        with self.open_output("delta_sigma_plot", wrapper=True, figsize=(5*nbin_lens, 4*nbin_source)) as fig:
            axes = fig.file.subplots(nbin_source, nbin_lens, squeeze=False)
            for s in range(nbin_source):
                for l in range(nbin_lens):
                    axes[s, l].set_title(f"Source {s}, Lens {l}")
                    axes[s, l].set_xlabel("Radius [arcmin]")
                    axes[s, l].set_ylabel(r"$\Delta \Sigma [M_\odot / pc^2]")
                    axes[s, l].grid()
                    x = sacc_data.get_tag("theta", tracers=(f"source_{s}", f"lens_{l}"))
                    y = sacc_data.get_mean(tracers=(f"source_{s}", f"lens_{l}"))
                    axes[s, l].plot(x, y)
            plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Plot in r coordinates
        nbin_source = sacc_data.metadata['nbin_source']
        nbin_lens = sacc_data.metadata['nbin_lens']
        with self.open_output("delta_sigma_r_plot", wrapper=True, figsize=(5*nbin_lens, 4*nbin_source)) as fig:
            axes = fig.file.subplots(nbin_source, nbin_lens, squeeze=False)
            for s in range(nbin_source):
                for l in range(nbin_lens):
                    axes[s, l].set_title(f"Source {s}, Lens {l}")
                    axes[s, l].set_xlabel("Radius [Mpc]")
                    axes[s, l].set_ylabel(r"$R \cdot \Delta \Sigma [M_\odot / pc^2]$")
                    axes[s, l].grid()
                    x = sacc_data.get_tag("r_mpc", tracers=(f"source_{s}", f"lens_{l}"))
                    y = sacc_data.get_mean(tracers=(f"source_{s}", f"lens_{l}"))
                    axes[s, l].plot(x, y * np.array(x))
            plt.subplots_adjust(hspace=0.3, wspace=0.3)