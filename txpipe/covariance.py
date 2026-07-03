from ceci.config import StageParameter
from .base_stage import PipelineStage
from .data_types import (
    HDFFile,
    FiducialCosmology,
    SACCFile,
)
import numpy as np

sq_deg_on_sky = 360**2 / np.pi


class TXFourierCovariance(PipelineStage):
    """
    Compute a Fourier-space covariance matrix using TJPCov.

    This stage delegates all covariance computation to TJPCov v0.5 via its
    CovarianceCalculator interface. The choice of covariance type (e.g.
    Gaussian f_sky, NaMaster, SSC) and all associated options are specified
    through the configuration.

    The stage reads cosmology, tracer metadata (number densities, shape noise,
    survey area) and the measured power spectra from earlier pipeline stages,
    builds a TJPCov configuration, and writes the resulting covariance back
    into the output sacc file.

    Common values for ``cov_type``:
        - ``FourierGaussianFsky``: Gaussian Knox-formula covariance using only
          the sky fraction.
        - ``FourierGaussianNmt``: Gaussian covariance using NaMaster coupling
          matrices (accounts for mask geometry).
        - ``FourierSSCHaloModel``: Super-sample covariance via a halo model.

    Multiple types can be combined to sum their contributions, e.g.::

        cov_type: [FourierGaussianFsky, FourierSSCHaloModel]
    """

    name = "TXFourierCovariance"
    parallel = False

    inputs = [
        ("fiducial_cosmology", FiducialCosmology),
        ("twopoint_data_fourier", SACCFile),
        ("tracer_metadata", HDFFile),
    ]

    outputs = [
        ("summary_statistics_fourier", SACCFile),
    ]

    config_options = {
        "cov_type": StageParameter(
            list,
            default=["FourierGaussianFsky"],
            msg="TJPCov covariance class name(s) to use.",
        ),
        "galaxy_bias": StageParameter(
            list,
            default=[],
            msg="Galaxy bias per lens bin. Defaults to 1.0 for every bin if empty.",
        ),
        "IA": StageParameter(
            float,
            default=0.0,
            msg="Intrinsic alignment amplitude passed to CCL WeakLensingTracer.",
        ),
        "outdir": StageParameter(
            str,
            default=".",
            msg="Directory for TJPCov to cache intermediate results.",
        ),
        "fsky": StageParameter(
            float,
            default=-1.0,
            msg="Sky fraction. If negative, it is computed from the survey area in tracer_metadata.",
        ),
        "use_true_shear": StageParameter(
            bool,
            default=False,
            msg="If True, set shape-noise sigma_e to zero (for noiseless simulations).",
        ),
    }

    def run(self):
        import sacc
        from tjpcov.covariance_calculator import CovarianceCalculator

        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl()

        sacc_file = sacc.Sacc.load_fits(self.get_input("twopoint_data_fourier"))
        sacc_file = self._filter_sacc(sacc_file, space="fourier")

        meta = self._read_tracer_metadata()
        tjp_config = self._build_tjpcov_config(cosmo, sacc_file, meta)

        calculator = CovarianceCalculator(tjp_config)
        cov = calculator.get_covariance()

        sacc_file.add_covariance(cov, overwrite=True)
        sacc_file.save_fits(
            self.get_output("summary_statistics_fourier"), overwrite=True
        )

    def _filter_sacc(self, sacc_file, space):
        import sacc as sacc_lib

        if space == "fourier":
            keep_types = [
                sacc_lib.standard_types.galaxy_shear_cl_ee,
                sacc_lib.standard_types.galaxy_shearDensity_cl_e,
                sacc_lib.standard_types.galaxy_density_cl,
            ]
        else:
            keep_types = [
                sacc_lib.standard_types.galaxy_density_xi,
                sacc_lib.standard_types.galaxy_shearDensity_xi_t,
                sacc_lib.standard_types.galaxy_shear_xi_plus,
                sacc_lib.standard_types.galaxy_shear_xi_minus,
            ]

        indices = np.concatenate([sacc_file.indices(dt) for dt in keep_types])
        sacc_file.keep_indices(indices)
        sacc_file.to_canonical_order()
        return sacc_file

    def _read_tracer_metadata(self):
        with self.open_input("tracer_metadata") as f:
            area_deg2 = f["tracers"].attrs["area"]
            area_unit = f["tracers"].attrs["area_unit"]
            if area_unit != "deg^2":
                raise ValueError(
                    f"Unexpected area unit '{area_unit}' in tracer_metadata. Expected 'deg^2'."
                )

            N_eff = f["tracers/N_eff"][:] if "tracers/N_eff" in f else np.array([])
            sigma_e = (
                f["tracers/sigma_e"][:] if "tracers/sigma_e" in f else np.array([])
            )
            N_lens = (
                f["tracers/lens_counts"][:] if "tracers/lens_counts" in f else np.array([])
            )

        area_arcmin2 = area_deg2 * 3600.0
        fsky = area_deg2 / sq_deg_on_sky

        return {
            "fsky": fsky,
            "area_arcmin2": area_arcmin2,
            "N_eff": N_eff,
            "sigma_e": sigma_e,
            "N_lens": N_lens,
        }

    def _build_tjpcov_config(self, cosmo, sacc_file, meta):
        fsky = self.config["fsky"]
        if fsky < 0:
            fsky = meta["fsky"]

        nbin_source = len(meta["N_eff"])
        nbin_lens = len(meta["N_lens"])
        area_arcmin2 = meta["area_arcmin2"]

        galaxy_bias = list(self.config["galaxy_bias"])
        if not galaxy_bias:
            galaxy_bias = [1.0] * nbin_lens

        if len(galaxy_bias) != nbin_lens:
            raise ValueError(
                f"galaxy_bias has {len(galaxy_bias)} entries but there are "
                f"{nbin_lens} lens bins."
            )

        sigma_e = meta["sigma_e"].copy()
        if self.config["use_true_shear"]:
            sigma_e = np.zeros(nbin_source)

        tjpcov_section = {
            "cov_type": self.config["cov_type"],
            "sacc_file": sacc_file,
            "cosmo": cosmo,
            "fsky": fsky,
            "IA": self.config["IA"],
            "outdir": self.config["outdir"],
        }

        for i in range(nbin_lens):
            tjpcov_section[f"bias_lens_{i}"] = galaxy_bias[i]
            tjpcov_section[f"Ngal_lens_{i}"] = (
                meta["N_lens"][i] / area_arcmin2
            )

        for i in range(nbin_source):
            tjpcov_section[f"sigma_e_source_{i}"] = float(sigma_e[i])
            tjpcov_section[f"Ngal_source_{i}"] = (
                meta["N_eff"][i] / area_arcmin2
            )

        return {"tjpcov": tjpcov_section}


class TXRealCovariance(TXFourierCovariance):
    """
    Compute a real-space covariance matrix using TJPCov.

    This stage delegates all covariance computation to TJPCov v0.5 via its
    CovarianceCalculator interface. The choice of covariance type and all
    associated options are specified through the configuration.

    The stage reads cosmology, tracer metadata and the measured correlation
    functions from earlier pipeline stages, builds a TJPCov configuration,
    and writes the resulting covariance back into the output sacc file.

    Common values for ``cov_type``:
        - ``RealGaussianFsky``: Gaussian covariance projected to real space
          using a Wigner transform, accounting only for the sky fraction.

    Multiple types can be combined to sum their contributions.
    """

    name = "TXRealCovariance"
    parallel = False

    inputs = [
        ("fiducial_cosmology", FiducialCosmology),
        ("twopoint_data_real", SACCFile),
        ("tracer_metadata", HDFFile),
    ]

    outputs = [
        ("summary_statistics_real", SACCFile),
    ]

    config_options = {
        "cov_type": StageParameter(
            list,
            default=["RealGaussianFsky"],
            msg="TJPCov covariance class name(s) to use.",
        ),
        "galaxy_bias": StageParameter(
            list,
            default=[],
            msg="Galaxy bias per lens bin. Defaults to 1.0 for every bin if empty.",
        ),
        "IA": StageParameter(
            float,
            default=0.0,
            msg="Intrinsic alignment amplitude passed to CCL WeakLensingTracer.",
        ),
        "outdir": StageParameter(
            str,
            default=".",
            msg="Directory for TJPCov to cache intermediate results.",
        ),
        "fsky": StageParameter(
            float,
            default=-1.0,
            msg="Sky fraction. If negative, it is computed from the survey area in tracer_metadata.",
        ),
        "use_true_shear": StageParameter(
            bool,
            default=False,
            msg="If True, set shape-noise sigma_e to zero (for noiseless simulations).",
        ),
    }

    def run(self):
        import sacc
        from tjpcov.covariance_calculator import CovarianceCalculator

        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl()

        sacc_file = sacc.Sacc.load_fits(self.get_input("twopoint_data_real"))
        sacc_file = self._filter_sacc(sacc_file, space="real")

        meta = self._read_tracer_metadata()
        tjp_config = self._build_tjpcov_config(cosmo, sacc_file, meta)

        calculator = CovarianceCalculator(tjp_config)
        cov = calculator.get_covariance()

        sacc_file.add_covariance(cov, overwrite=True)
        sacc_file.save_fits(
            self.get_output("summary_statistics_real"), overwrite=True
        )
