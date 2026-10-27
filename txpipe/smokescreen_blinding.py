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
        "encrypt_unblinded": StageParameter(
            bool, False, msg="Whether to save an encrypted copy of the unblinded data, and its key."
        ),
        "encryption_dir": StageParameter(
            str, "", msg="Directory for the encrypted data and key. Empty means next to the unblinded data."
        ),
        "delete_unblinded": StageParameter(bool, False, msg="Whether to delete unblinded data after blinding."),
    }

    def run(self):
        """
        Run the analysis for this stage.

         - Load the unblinded two-point sacc file
         - Select the points the likelihood models
         - Blind them with Smokescreen
         - Save the blinded data
         - Optionally encrypt and/or delete the unblinded data
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

        modelled_sack, modelled_indices = self.select_modelled_data(sack, likelihood_path)
        concealed_vector = self.conceal(modelled_sack, likelihood_path)

        # Put the blinded values into a copy of the full data, so that the
        # covariance, tracers, window functions, and any unmodelled points
        # are kept as they were.
        blinded_sack = sack.copy()
        for i, value in zip(modelled_indices, concealed_vector):
            blinded_sack.data[i].value = value

        import smokescreen

        blinded_sack.metadata["concealed"] = True
        blinded_sack.metadata["blinding_method"] = "smokescreen"
        blinded_sack.metadata["smokescreen_version"] = smokescreen.__version__
        blinded_sack.metadata["blinding_likelihood"] = os.path.basename(likelihood_path)
        blinded_sack.save_fits(self.get_output("twopoint_data_blinded"), overwrite=True)

        print("Congratulations you are now blind.")

        if self.config["encrypt_unblinded"]:
            self.encrypt_unblinded(unblinded_fname)

        # Optionally make sure we stay blind by deleting the pre-blinding
        # data. We leave an empty file so the pipeline still sees it.
        if self.config["delete_unblinded"]:
            print(f"Replacing {unblinded_fname} with empty file.")
            open(unblinded_fname, "w").close()

    def encrypt_unblinded(self, unblinded_fname):
        """
        Save an encrypted copy of the unblinded data with Smokescreen,
        so it can be recovered later with the key.
        """
        from smokescreen.encryption import encrypt_file

        encryption_dir = self.config["encryption_dir"] or os.path.dirname(unblinded_fname)
        encrypt_file(unblinded_fname, path_to_save=encryption_dir, save_file=True, keep_original=True)

        # This is the naming scheme Smokescreen uses
        stem = os.path.join(encryption_dir, os.path.basename(unblinded_fname).split(".")[0])
        print(f"Saved encrypted unblinded data to {stem}.encrpt")
        print(f"Warning: move the key {stem}.key somewhere safe, away from the data, to stay blind.")

    def conceal(self, modelled_sack, likelihood_path):
        """
        Compute the blinded data vector with Smokescreen.

        Parameters
        ----------
        modelled_sack: sacc.Sacc
            The data to blind, containing only points the likelihood models

        likelihood_path: str
            The firecrown likelihood file

        Returns
        -------
        array
            The blinded data vector, in the same order as modelled_sack
        """
        from smokescreen import ConcealDataVector
        from smokescreen.utils import load_cosmology_from_partial_dict
        import numpy as np

        cosmo = load_cosmology_from_partial_dict(self.config["reference_cosmology"])

        # YAML gives us lists, but Smokescreen expects ranges to be tuples
        shifts = {
            name: tuple(value) if isinstance(value, (list, tuple)) else value
            for name, value in self.config["shifts_dict"].items()
        }
        if not shifts:
            raise ValueError("No shifts_dict was set for SmokescreenBlinding, so nothing would be blinded")

        blinder = ConcealDataVector(
            cosmo,
            likelihood_path,
            shifts,
            modelled_sack,
            self.get_systematics(modelled_sack, cosmo),
            seed=self.config["seed"],
            shift_distr=self.config["shift_distribution"],
        )

        # We built modelled_sack so that the likelihood uses its points in order.
        indices = np.array(blinder.likelihood.get_sacc_indices())
        if not np.array_equal(indices, np.arange(len(modelled_sack))):
            raise RuntimeError("The likelihood did not use the blinding data points in the expected order")

        blinder.calculate_concealing_factor(factor_type=self.config["shift_type"])
        return blinder.apply_concealing_to_likelihood_datavec()

    def get_systematics(self, modelled_sack, cosmo):
        """
        Get the systematics parameter values for the likelihood.

        Returns None if Smokescreen should use the likelihood's defaults.
        """
        from .utils.theory import make_bias_parameters

        systematics = self.config["systematics"]
        if systematics:
            return {name: float(value) for name, value in systematics.items()}

        # As in TXBlinding, the TXPipe model can use a bias b0 / growth
        if self.using_default_model() and self.config["b0"] != 0:
            return make_bias_parameters(self.config["b0"], modelled_sack, cosmo)

        return None

    def select_modelled_data(self, sack, likelihood_path):
        """
        Make a copy of the data containing only the points that the
        likelihood models, in the order the likelihood uses them.

        Smokescreen checks that the sacc data it is given matches the
        likelihood's data vector exactly. TXPipe two-point files can contain
        points the likelihood does not model, like B-modes in harmonic space
        or points removed by scale cuts, so we cut these out here. They are
        left unchanged (unblinded) in the output.

        Parameters
        ----------
        sack: sacc.Sacc
            The full unblinded data

        likelihood_path: str
            The firecrown likelihood file

        Returns
        -------
        modelled_sack: sacc.Sacc
            A copy of the data with only the modelled points

        modelled_indices: array
            The index in the full data of each point in modelled_sack
        """
        from firecrown.likelihood import load_likelihood, NamedParameters
        import numpy as np

        likelihood, _ = load_likelihood(likelihood_path, NamedParameters({"sacc_data": sack.copy()}))
        modelled_indices = np.array(likelihood.get_sacc_indices())

        modelled_sack = sack.copy()
        modelled_sack.reorder(modelled_indices)

        # The likelihood's sacc indices must refer to the input file. They will
        # not if it removes points from the sacc data, e.g. to make scale cuts.
        if not np.array_equal(modelled_sack.mean, likelihood.get_data_vector()):
            raise ValueError(
                f"The data vector of likelihood {likelihood_path} does not match the "
                "input data at its sacc indices. If it removes points from the sacc "
                "data (for example for scale cuts) it should select them instead, "
                "so that the indices still refer to the original file."
            )

        # The TXPipe theory model always uses a dummy covariance, because the
        # measured one may be missing (harmonic space) or not positive-definite
        # (real space jackknife). The covariance does not affect the blinding
        # shift, but Smokescreen checks it matches the likelihood's, so we use
        # the same dummy one here. Other likelihoods use the data covariance.
        if self.using_default_model():
            modelled_sack.add_covariance(np.ones(len(modelled_sack)), overwrite=True)

        n_unmodelled = len(sack) - len(modelled_indices)
        if n_unmodelled:
            unmodelled = np.setdiff1d(np.arange(len(sack)), modelled_indices)
            data_types = sorted(set(sack.data[i].data_type for i in unmodelled))
            print(
                f"Warning: {n_unmodelled} of {len(sack)} data points are not modelled by the "
                f"likelihood and will NOT be blinded. Data types: {', '.join(data_types)}"
            )

        return modelled_sack, modelled_indices

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
