from .base_stage import PipelineStage
from .data_types import SACCFile
from .utils.theory import default_theory_model
import os
from ceci.config import StageParameter


class SmokescreenBlinding(PipelineStage):
    """
    Blinds two-point measurements using Smokescreen

    Smokescreen (https://github.com/LSSTDESC/Smokescreen) conceals a data vector
    by shifting it by the difference between theory predictions at a reference
    cosmology and at a randomly shifted one (Muir et al. 2020).

    The theory is computed by a firecrown likelihood. By default this is the
    TXPipe 3x2pt theory model, but any firecrown likelihood file can be given
    with the likelihood_path option. The stage works on any sacc file the
    likelihood can model, in real or harmonic space; use pipeline aliases to
    connect the generic twopoint_data_raw and twopoint_data_blinded tags to
    the files you want to blind.

    The configuration options follow those of the Smokescreen command line tool.
    """

    name = "SmokescreenBlinding"
    parallel = False
    inputs = [
        ("twopoint_data_raw", SACCFile),
    ]
    outputs = [
        ("twopoint_data_blinded", SACCFile),
    ]
    config_options = {
        "likelihood_path": StageParameter(
            str, "", msg="Path to a firecrown likelihood file. If empty, the TXPipe 3x2pt theory model is used."
        ),
        "reference_cosmology": StageParameter(
            dict,
            {},
            msg="Reference cosmology parameters (CCL names). Unset parameters take CCL vanilla LCDM values.",
        ),
        "shifts_dict": StageParameter(
            dict,
            {},
            msg="Parameter shifts: {param: value} for a fixed value, or {param: [a, b]} to draw it at random.",
        ),
        "shift_distribution": StageParameter(
            str, "flat", msg="Distribution of random shifts: 'flat' (U(a, b)) or 'gaussian' (N(a, b))."
        ),
        "shift_type": StageParameter(str, "add", msg="How the shift is applied: 'add' or 'mult'."),
        "systematics": StageParameter(
            dict, {}, msg="Values for the likelihood's systematics parameters. Empty uses their defaults."
        ),
        "b0": StageParameter(
            float,
            0.0,
            msg="TXPipe theory model only, if systematics is empty: lens bias b0/growth. Zero means unset.",
        ),
        "seed": StageParameter(str, "2112", msg="Seed that uniquely specifies the shifts."),
    }

    def run(self):
        """
        Run the analysis for this stage.

         - Load the unblinded two-point sacc file
        """
        import sacc

        unblinded_fname = self.get_input("twopoint_data_raw")

        # We may have blinded already. The error message would be quite
        # obscure, so check here and say something more straightforward.
        size = os.stat(unblinded_fname).st_size
        if size == 0:
            raise ValueError(
                "The raw 2point file you specified has zero size. "
                "Maybe you already run the blinding stage? "
                "It can clear the input file to enforce blindness."
            )

        sack = sacc.Sacc.load_fits(unblinded_fname)
        likelihood_path = self.get_likelihood_path()
        print(f"Blinding {unblinded_fname} with Smokescreen, likelihood {likelihood_path}")

    def get_likelihood_path(self):
        """
        Return the path to the firecrown likelihood file to use.
        """
        path = self.config["likelihood_path"]
        if not path:
            return default_theory_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"Likelihood file {path} not found")
        return path

    def using_default_model(self):
        return self.get_likelihood_path() == default_theory_model
