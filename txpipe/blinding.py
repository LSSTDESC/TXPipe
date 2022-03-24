from .base_stage import PipelineStage
from .data_types import SACCFile
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
        blinded_sack = self.blind_muir(sack)

        # Save
        blinded_sack.save_fits(self.get_output("twopoint_data_real"), overwrite=True)

        # Optionally make sure we stay blind by deleting the pre-blinding
        # file.
        if self.config["delete_unblinded"]:
            print(f"Replacing {unblinded_fname} with empty file.")
            open(unblinded_fname, "w").close()

    def blind_muir(self, sack):
        # Get the parameters, fiducial and offset.
        # The signature is a short sequence that means we can
        # check that it is unchanged
        signature, fid_params, offset_params = self.get_parameters()

        # Save the signature into the output
        sack.metadata["blinding_signature"] = signature

        # Turn the b0 parameter into the b(z) per lens bin.
        bias = self.compute_bias(sack, fid_params)

        # Get the configuration dictionary
        firecrown_config, types = self.make_firecrown_config(sack, bias)

        ## now try to get predictions
        print("Computing fiducial theory")
        fid_theory = self.compute_theory_vector(firecrown_config, fid_params, types)
        print("Computing offset theory")
        offset_theory = self.compute_theory_vector(
            firecrown_config, offset_params, types
        )

        # Get the parameter shift vector
        diff_vec = offset_theory - fid_theory

        # And add this to the original data.
        for p, delta in zip(sack.data, diff_vec):
            p.value += delta

        print(f"Congratulations you are now blind.")
        return sack

    def get_parameters(self):
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

        return signature, fid_params, offset_params

    def compute_bias(self, sack, fid_params):
        import pyccl as ccl

        print("Computing bias values b(z)")
        # now get bias as a function of redshift for each lens bin
        bias = {}
        fid_cosmo = ccl.Cosmology(**fid_params)
        b0 = self.config["b0"]
        for key, tracer in sack.tracers.items():
            if "lens" in key:
                z_eff = (tracer.z * tracer.nz).sum() / tracer.nz.sum()
                a_eff = 1 / (1 + z_eff)
                bias[key] = b0 / ccl.growth_factor(fid_cosmo, a_eff)
        return bias

    def make_firecrown_config(self, sack, bias):
        from sacc.utils import unique_list

        # We will run firecrown with the old and new parameters,
        # but otherwise the same configuration
        firecrown_config = {
            "parameters": {"Omega_k": 0.0, "wa": 0.0, "one": 1},
            "two_point": {
                "module": "firecrown.ccl.two_point",
                "sacc_data": sack,
                "systematics": {"dummy": {"kind": "PhotoZShiftBias", "delta_z": "one"}},
                "sources": {},
                "statistics": {},
            },
        }

        for k, v in bias.items():
            firecrown_config["parameters"][f"bias_{k}"] = v

        # Tell FireCrown to use the sources we have here, and
        # their types.
        firecrown_sources = firecrown_config["two_point"]["sources"]
        for key, tracer in sack.tracers.items():
            # Assume the word source or lens will be in the name.
            # Bit of a hack.
            if "source" in key:
                firecrown_sources[key] = {"kind": "WLSource", "sacc_tracer": key}

            if "lens" in key:
                firecrown_sources[key] = {
                    "kind": "NumberCountsSource",
                    "bias": f"bias_{key}",
                    "sacc_tracer": key,
                }

        ## Make a list of the unique tracer/bin groups
        types = []
        for p in sack.data:
            name = f"{p.data_type}_{p.tracers[0]}_{p.tracers[1]}"
            types.append((p.data_type, p.tracers, name))

        types = unique_list(types)

        # Collect together the statistics we will need
        firecrown_stats = firecrown_config["two_point"]["statistics"]
        for dtype, (tracer1, tracer2), name in types:
            firecrown_stats[name] = {
                "sources": [tracer1, tracer2],
                "sacc_data_type": dtype,
            }

        return firecrown_config, types

    def compute_theory_vector(self, config, params, types):
        import firecrown

        # Set up the firecrown configuration
        config["parameters"].update(params)
        config2, data = firecrown.parse(config)

        # Run the cosmology.  The loglike also as a by-product
        # fills in the predictions deep inside the data dictionary.
        cosmo = firecrown.get_ccl_cosmology(config2["parameters"])
        firecrown.compute_loglike(cosmo=cosmo, data=data)

        # Pull out and stack the theory predictions at this point
        results = data["two_point"]["data"]["statistics"]
        return np.hstack(
            [results[stat_name].predicted_statistic_ for (_, _, stat_name) in types]
        )


class TXNullBlinding(PipelineStage):
    """
    Pretend to blind but actually do nothing.

    This null stage trivially copies the real raw data to real without blinding,
    for use with simulated data, etc.
    """

    name = "TXNullBlinding"
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
