from .base_stage import PipelineStage
from .data_types import SACCFile
from .utils.theory import theory_3x2pt
import numpy as np
import warnings
import os


class TXBlinding(PipelineStage):
    """
    Blinds real-space measurements following Muir et al

    This stage computes the shift in theory predictions between a fiducial cosmology
    and a cosmology randomly chosen based on a seed value.

    It then applies that shift to the observations in a sacc file.
    """

    name = "TXBlinding"
    parallel = False
    inputs = [
        ("twopoint_data_real_raw", SACCFile),
    ]
    outputs = [
        ("twopoint_data_real", SACCFile),
    ]
    config_options = {
        "seed": 1972,  ## seed uniquely specifies the shift in parameters
        "Omega_b": [0.0485, 0.001],  ## fiducial_model_value, shift_sigma
        "Omega_c": [0.2545, 0.01],
        "w0": [-1.0, 0.1],
        "h": [0.682, 0.02],
        "sigma8": [0.801, 0.01],
        "n_s": [0.971, 0.03],
        "b0": 0.95,  ### we assume bias to be of the form b0/growth
        "delete_unblinded": False,
    }

    def run(self):
        """
        Run the analysis for this stage.

         - Load two point SACC file
         - Blinding it
         - Output blinded data
         - Optionally deletete unblinded data
        """
        import pyccl
        import firecrown
        import sacc

        unblinded_fname = self.get_input("twopoint_data_real_raw")

        # We may have blinded already. The error message would be quite
        # obscure, so check here and say something more straightforward.
        size = os.stat(unblinded_fname).st_size
        if size == 0:
            raise ValueError(
                "The raw 2point file you specified has zero size. "
                "Maybe you already run the blinding stage? "
                "It clears the input file to enforce blindness."
            )

        # Load
        sack = sacc.Sacc.load_fits(unblinded_fname)

        # Blind
        self.blind_muir(sack)

        # Save
        sack.save_fits(self.get_output("twopoint_data_real"), overwrite=True)

        # Optionally make sure we stay blind by deleting the pre-blinding
        # file.
        if self.config["delete_unblinded"]:
            print(f"Replacing {unblinded_fname} with empty file.")
            open(unblinded_fname, "w").close()

    def blind_muir(self, sack):
        # Get the parameters, fiducial and offset.
        # The signature is a short sequence that means we can
        # check that it is unchanged
        signature, fid_cosmo, offset_cosmo = self.get_parameters()

        # Save the signature into the output
        sack.metadata["blinding_signature"] = signature


        ## now try to get predictions
        print("Computing fiducial theory")
        fid_theory = theory_3x2pt(fid_cosmo, sack, bias=self.config["b0"])

        print("Computing offset theory")
        offset_theory = theory_3x2pt(offset_cosmo, sack, bias=self.config["b0"])

        # Get the parameter shift vector
        diff_vec = offset_theory.get_mean() - fid_theory.get_mean()

        # And add this to the original data.
        for p, delta in zip(sack.data, diff_vec):
            p.value += delta

        print(f"Congratulations you are now blind.")

    def get_parameters(self):
        import pyccl
        seed = self.config["seed"]
        print(f"Blinding with seed {seed}")

        # This is the legacy random number generator
        # which will always produce the same values
        rng = np.random.RandomState(seed=seed)

        # blind signature -- this ensures seed is consistent across
        # numpy versions.  We do this before and after
        signature_bytes = rng.bytes(4)
        signature = "".join(format(x, "02x") for x in signature_bytes)

        if self.rank == 0:
            print(f"Blinding signature: {signature}")

        # Pull out the fiducial parameters from the config
        fid_params = {
            p: self.config[p][0]
            for p in ["Omega_b", "Omega_c", "h", "w0", "sigma8", "n_s"]
        }

        # Get the parameters in the offset space
        offset_params = fid_params.copy()

        # This should have a standard order since python dictionaries
        # now maintain their order.
        for par in fid_params.keys():
            offset_params[par] += self.config[par][1] * rng.normal(0.0, 1.0)

        fid_cosmo = pyccl.Cosmology(**fid_params)
        offset_cosmo = pyccl.Cosmology(**offset_params)

        return signature, fid_cosmo, offset_cosmo




class TXNullBlinding(PipelineStage):
    """
    Pretend to blind but actually do nothing.

    This null stage trivially copies the real raw data to real without blinding,
    for use with simulated data, etc.
    """

    name = "TXNullBlinding"
    parallel = False
    inputs = [
        ("twopoint_data_real_raw", SACCFile),
    ]
    outputs = [
        ("twopoint_data_real", SACCFile),
    ]
    config_options = {}

    def run(self):
        """
        Run the analysis for this stage.

         - Load two point SACC file
         - Copy two point SACC file twopoint_data_real_raw to twopoint_data_real
        """
        import shutil

        unblinded_fname = self.get_input("twopoint_data_real_raw")
        shutil.copyfile(unblinded_fname, self.get_output("twopoint_data_real"))
