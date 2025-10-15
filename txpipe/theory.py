from .base_stage import PipelineStage
from .data_types import FiducialCosmology, SACCFile
from .utils.theory import theory_3x2pt
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
        "galaxy_bias": StageParameter(list, [0.0], msg="Galaxy bias values per bin, [0.0] for unit bias, or single negative value for global bias parameter"),
        "smooth": StageParameter(bool, False, msg="Whether to smooth the theory predictions"),
    }

    def run(self):
        import sacc

        filename = self.get_input("twopoint_data_real")
        s = sacc.Sacc.load_fits(filename)

        # TODO: when there is a better Cosmology serialization method
        # switch to that
        print("Manually specifying matter_power_spectrum and Neff")
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl(
            matter_power_spectrum="halofit", Neff=3.046
        )

        print(cosmo)

        # We let the user specify bias values in one of three
        # ways.
        # 1) leaving the default [0.0] - this indicates unit bias
        # 2) specifying a single negative value - this means to use a single
        #    global bias parameter with each bin's bias given by b / D(z)
        # 3) a bias parameter per bin
        # I know version (2) is confusing, but we want the result to work
        # even when there's only a single bin.
        bias = self.config['galaxy_bias']
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
        "galaxy_bias": StageParameter(list, [0.0], msg="Galaxy bias values per bin, [0.0] for unit bias, or single negative value for global bias parameter"),
        "smooth": StageParameter(bool, False, msg="Whether to smooth the theory predictions"),
    }
    
    def run(self):
        import sacc

        filename = self.get_input("twopoint_data_fourier")
        s = sacc.Sacc.load_fits(filename)

        # TODO: when there is a better Cosmology serialization method
        # switch to that
        print("Manually specifying matter_power_spectrum and Neff")
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl(
            matter_power_spectrum="halofit", Neff=3.046
        )
        print(cosmo)

        s_theory = theory_3x2pt(cosmo, s, smooth=self.config["smooth"])

        # Remove covariance
        s_theory.covariance = None

        # save the output to Sacc file
        s_theory.save_fits(self.get_output("twopoint_theory_fourier"), overwrite=True)