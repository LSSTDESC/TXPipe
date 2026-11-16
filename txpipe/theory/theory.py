from ..base_stage import PipelineStage
from ..data_types import FiducialCosmology, SACCFile
from .theory_utils import theory_3x2pt
from ceci.config import StageParameter
import numpy as np


class TXTwoPointTheoryReal(PipelineStage):
    """
    Compute theory predictions for real-space 3x2pt measurements.
    Uses CCL in real space and saves to a sacc file.
    """

    name = "TXTwoPointTheoryReal"
    parallel = False
    inputs = [
        ("twopoint_data_real", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),  # For example lines
    ]
    outputs = [
        ("twopoint_theory_real", SACCFile),
    ]

    config_options = {
        "galaxy_bias": StageParameter(
            list,
            [0.0],
            msg="Galaxy bias values per bin, [0.0] for unit bias, or single negative value for global bias parameter",
        ),
        "smooth": StageParameter(bool, False, msg="Whether to smooth the n(z) for theory predictions"),
    }

    def run(self):
        import sacc

        filename = self.get_input("twopoint_data_real")
        s = sacc.Sacc.load_fits(filename)

        # TODO: when there is a better Cosmology serialization method
        # switch to that
        print("Manually specifying matter_power_spectrum and Neff")
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl(matter_power_spectrum="halofit", Neff=3.046)

        print(cosmo)

        # We let the user specify bias values in one of three
        # ways.
        # 1) leaving the default [0.0] - this indicates unit bias
        # 2) specifying a single negative value - this means to use a single
        #    global bias parameter with each bin's bias given by b / D(z)
        # 3) a bias parameter per bin
        # I know version (2) is confusing, but we want the result to work
        # even when there's only a single bin.
        bias = self.config["galaxy_bias"]
        if bias == [0.0]:
            bias = None
        elif len(bias) == 1 and bias[0] < 0:
            bias = -bias[0]
        s_theory = theory_3x2pt(cosmo, s, bias=bias, smooth=self.config["smooth"])

        # Remove covariance
        s_theory.covariance = None

        # save the output to Sacc file
        s_theory.save_fits(self.get_output("twopoint_theory_real"), overwrite=True)


class TXTwoPointTheoryFourier(TXTwoPointTheoryReal):
    """
    Compute theory predictions for Fourier-space 3x2pt measurements.
    Also uses CCL and saves to a sacc file
    """

    name = "TXTwoPointTheoryFourier"
    parallel = False
    inputs = [
        ("twopoint_data_fourier", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),  # For example lines
    ]
    outputs = [
        ("twopoint_theory_fourier", SACCFile),
    ]

    config_options = {
        "galaxy_bias": StageParameter(
            list,
            [0.0],
            msg="Galaxy bias values per bin, [0.0] for unit bias, or single negative value for global bias parameter",
        ),
        "smooth": StageParameter(bool, False, msg="Whether to smooth the n(z) for theory predictions"),
    }

    def run(self):
        import sacc

        filename = self.get_input("twopoint_data_fourier")
        s = sacc.Sacc.load_fits(filename)

        # TODO: when there is a better Cosmology serialization method
        # switch to that
        print("Manually specifying matter_power_spectrum and Neff")
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl(matter_power_spectrum="halofit", Neff=3.046)
        print(cosmo)

        # Same bias options as TXTwoPointTheoryReal
        bias = self.config["galaxy_bias"]
        if bias == [0.0]:
            bias = None
        elif len(bias) == 1 and bias[0] < 0:
            bias = -bias[0]
        s_theory = theory_3x2pt(cosmo, s, bias=bias, smooth=self.config["smooth"])

        # Remove covariance
        s_theory.covariance = None

        # save the output to Sacc file
        s_theory.save_fits(self.get_output("twopoint_theory_fourier"), overwrite=True)


class TXTwoPointTheoryFourierCCL(PipelineStage):
    """
    Theory predictions for the Fourier-space galaxy clustering spectra, computed with CCL directly.

    For every galaxy_density_cl data point in the measured sacc file this computes
    C_b = sum_ell W_b(ell) C_ell, where W_b is the NaMaster bandpower window stored with that
    point (used as it is, not renormalised) and C_ell is CCL's Limber angular power spectrum for
    two NumberCountsTracers with the lens n(z) from the sacc file and a bias that is constant in
    redshift within each bin (no RSD, no magnification).

    This replaces TXTwoPointTheoryFourier for clustering-only runs, which goes through Firecrown
    and has two problems: TXPipe's make_bias_parameters sets alphaz = 1, z_piv = 1, so the
    "constant" bias is really b * (1 + z) / 2; and Firecrown divides each bandpower window by
    its sum (and sets ell = 0, 1 to zero), which is wrong for NaMaster windows at low ell.

    Only lens (galaxy_density_cl) spectra are supported; anything else is an error.
    """

    name = "TXTwoPointTheoryFourierCCL"
    parallel = False
    inputs = [
        ("twopoint_data_fourier", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [
        ("twopoint_theory_fourier", SACCFile),
    ]

    config_options = {
        "galaxy_bias": StageParameter(
            list, [0.0], msg="Constant linear bias per lens bin; [0.0] for unit bias in every bin"
        ),
        "matter_power_spectrum": StageParameter(str, "halofit", msg="CCL nonlinear matter power spectrum"),
    }

    def run(self):
        import pyccl as ccl
        import sacc

        data_type = "galaxy_density_cl"
        s = sacc.Sacc.load_fits(self.get_input("twopoint_data_fourier"))
        other = [dt for dt in s.get_data_types() if dt != data_type]
        if other:
            raise ValueError(f"{self.name} only handles {data_type}, but the file also has {other}")

        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl(
            matter_power_spectrum=self.config["matter_power_spectrum"]
        )
        print(cosmo)

        lens = sorted(t for t in s.tracers if t.startswith("lens_"))
        bias = self.config["galaxy_bias"]
        if bias == [0.0]:
            bias = [1.0] * len(lens)
        if len(bias) != len(lens):
            raise ValueError(f"galaxy_bias has {len(bias)} values but there are {len(lens)} lens bins")

        tracers = {}
        for name in lens:
            t = s.tracers[name]
            b = bias[int(name.split("_")[1])]
            print(f"{name}: constant bias {b}")
            tracers[name] = ccl.NumberCountsTracer(
                cosmo, has_rsd=False, dndz=(t.z, t.nz), bias=(t.z, np.full_like(t.z, b, dtype=float))
            )

        s_theory = s.copy()
        s_theory.covariance = None
        for d in s_theory.data:
            d.value = 0.0
        for t1, t2 in s.get_tracer_combinations(data_type):
            ind = s.indices(data_type, (t1, t2))
            window = s.get_bandpower_windows(ind)
            if window is None:
                raise ValueError(f"No bandpower windows for {t1} x {t2}")
            ell = window.values.astype(int)
            cl = ccl.angular_cl(cosmo, tracers[t1], tracers[t2], ell)
            binned = window.weight.T @ cl  # weight is (n_ell, n_bandpowers)
            for i, v in zip(ind, binned):
                s_theory.data[i].value = v
            print(f"{t1} x {t2}: {len(ind)} bandpowers, ell {ell.min()}-{ell.max()}")

        s_theory.save_fits(self.get_output("twopoint_theory_fourier"), overwrite=True)
