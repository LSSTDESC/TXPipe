import pathlib
import numpy as np
import warnings
import yaml
import pathlib

# same convention as elsewhere
SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

default_theory_model = str(pathlib.Path(__file__).parents[0] / "theory_model.py")


def ccl_read_yaml(filename, **kwargs):
    import pyccl as ccl

    """Read the parameters from a YAML file.

    Args:
        filename (:obj:`str`) Filename to read parameters from.
    """
    with open(filename, "r") as fp:
        params = yaml.load(fp, Loader=yaml.Loader)

    # Now we assemble an init for the object since the CCL YAML has
    # extra info we don't need and different formatting.
    inits = dict(
        Omega_c=params["Omega_c"],
        Omega_b=params["Omega_b"],
        h=params["h"],
        n_s=params["n_s"],
        sigma8=None if params["sigma8"] == "nan" else params["sigma8"],
        A_s=None if params["A_s"] == "nan" else params["A_s"],
        Omega_k=params["Omega_k"],
        Neff=params["Neff"],
        w0=params["w0"],
        wa=params["wa"],
        bcm_log10Mc=params["bcm_log10Mc"],
        bcm_etab=params["bcm_etab"],
        bcm_ks=params["bcm_ks"],
        mu_0=params["mu_0"],
        sigma_0=params["sigma_0"],
    )
    if "z_mg" in params:
        inits["z_mg"] = params["z_mg"]
        inits["df_mg"] = params["df_mg"]

    if "m_nu" in params:
        inits["m_nu"] = params["m_nu"]
        inits["m_nu_type"] = "list"

    inits.update(kwargs)

    return ccl.Cosmology(**inits)


def fill_empty_sacc(sacc_data, ell_values=None, theta_values=None):
    """
    Make a sacc object containing
    """
    if (ell_values is None) and (theta_values is None):
        raise ValueError("Supplied an empty sacc file but no ell or theta values to fill it")
    elif (ell_values is not None) and (theta_values is not None):
        raise ValueError(
            "Supplied an empty sacc file and both theta and ell values to fill it. Just pick one"
        )


    is_fourier = ell_values is not None

    for t1 in list(sacc_data.tracers.keys()):
        t1_is_source = t1.startswith("source")
        t1_is_lens = t1.startswith("lens")
        index1 = t1.split("_")[1]
        if not (t1_is_source or t2_is_source):
            print(f"Skipping mock data for tracer {t1}")
            continue
        for t2 in sacc_data.tracers.keys():
            t2_is_source = t2.startswith("source")
            t2_is_lens = t2.startswith("lens")
            index2 = t2.split("_")[1]
            if not (t1_is_source or t2_is_source):
                continue

            if t1_is_source and t2_is_source:
                if is_ell:
                    dts = ["galaxy_shear_cl_ee"]
                else:
                    dts = ["galaxy_shear_xi_plus", "galaxy_shear_xi_minus"]

                if index2 > index1:
                    continue
            elif t1_is_source and t2_is_lens:
                if is_ell:
                    dts = ["galaxy_shearDensity_cl_e"]
                else:
                    dts = ["galaxy_shearDensity_xi_t"]
            else:
                if is_ell:
                    dts = ["galaxy_density_cl"]
                else:
                    dts = ["galaxy_density_xi"]

                if index2 > index1:
                    continue

            for dt in dts:
                if is_fourier:
                    for ell in ell_values:
                        sacc_data.add_data_point(dt, (t1, t2), 0.0, ell=ell)
                else:
                    for theta in theta_values:
                        sacc_data.add_data_point(dt, (t1, t2), 0.0, theta=theta)


def theory_3x2pt(
    cosmo, sacc_data, bias=None, smooth=False, ell_values=None, theta_values=None,
    theory_model=default_theory_model
):
    """
    Use FireCrown to generate the theory predictions for the data
    in a sacc file.

    If the supplied sacc data has only tracers and no data points then
    the output will be given data points at the ell or theta
    values supplued in the keyword options.

    Linear galaxy bias parameters can optionally be provided as an array.
    If left as None they are set to unity.

    Parameters
    ----------
    cosmo: pyccl.Cosmology
        The cosmology at which to get the theory predictions
    sacc_data: sacc.Sacc or str
        Sacc object or a path to one. If the file has only tracers and
        no data then ell_values or theta_values must be supplied.
    bias: float, array, or None
        If a float b_0, use the scheme b = b_0 / D(<z>) for each bin.
        If an array, use the given value for each bin
    ell_values: array, optional
        An array of ell's to use if no data points are in the sacc file
    theta_values: array, optional
        An array of theta's to use if no data points are in the sacc file

    Returns
    -------
    sacc_theory: sacc.Sacc
        A copy of the input sacc_data but with values
        replaced with theory predictions.
    """
    from firecrown.likelihood.likelihood import load_likelihood
    import sacc
    import pathlib

    # Make sure we have computed P(k, z). If this
    # has already been calculated then this doesn't
    # do so again.
    cosmo.compute_nonlin_power()

    # Read the sacc data file if needed
    if isinstance(sacc_data, (str, pathlib.Path)):
        sacc_data = sacc.Sacc.load_fits(sacc_data)
    else:
        sacc_data = sacc_data.copy()

    # The user can pass in an empty sacc file, with no data points in,
    # if they also pass in either ell or theta values to fill it with.
    # They can pass in either but not both.
    if len(sacc_data.data) == 0:
        fill_empty_sacc(sacc_data, ell_values=ell_values, theta_values=theta_values)

    # Use the FireCrown machinery to compute the likelihood and as
    # a by-product the theory
    build_parameters = {
        "sacc_data": sacc_data,
        "bias": bias,
        "smooth": smooth,
    }

    # These stages are a copy of what is done inside the FireCrown connectors
    likelihood, tools = load_likelihood(theory_model, build_parameters)
    tools.prepare(cosmo)
    loglike = likelihood.compute_loglike(tools)

    sacc_theory = sacc_data.copy()

    # Set everything to zero first in case there are BB measurements
    for d in sacc_theory.data:
        d.value = 0.0

    # Fill in the values of the computed theory
    for i, v in zip(tools._tx_indices, likelihood.predicted_data_vector):
        sacc_theory.data[i].value = v

    return sacc_theory


