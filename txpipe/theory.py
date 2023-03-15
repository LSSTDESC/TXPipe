from .base_stage import PipelineStage
from .data_types import FiducialCosmology, SACCFile
from .utils.theory import theory_3x2pt
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
        "galaxy_bias": [0.0],
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
        bias = self.config['galaxy_bias']
        if bias == [0.0]:
            bias = None
        elif len(bias) == 1 and bias[0] < 0:
            bias = -bias[0]
        s_theory = theory_3x2pt(cosmo, s, bias=bias)

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
        "galaxy_bias": [0.0],
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

        s_theory = theory_3x2pt(s, cosmo)

        # Remove covariance
        s_theory.covariance = None

        # save the output to Sacc file
        s_theory.save_fits(self.get_output("twopoint_theory_fourier"), overwrite=True)
