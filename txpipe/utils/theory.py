import pathlib
import numpy as np
import warnings
import yaml

# same convention as elsewhere
SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2


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
    cosmo, sacc_data, bias=None, smooth=False, ell_values=None, theta_values=None
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
    import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
    import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
    from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
    import sacc
    import pathlib

    if isinstance(sacc_data, (str, pathlib.Path)):
        sacc_data = sacc.Sacc.load_fits(sacc_data)
    else:
        sacc_data = sacc_data.copy()

    # If someone explains to me how to have fixed linear bias systematic
    # as parameters instead of passing it in like this then that would
    # be better here.
    if isinstance(bias, float):
        bias = compute_fiducial_bias(bias, sacc_data, cosmo)
    elif bias is None:
        bias = np.ones(len(sacc_data.tracers))

    # We can optionally smooth the n(z). This helped in Prat et al.
    if smooth:
        smooth_sacc_nz(sacc_data)

    # The user can pass in an empty sacc file, with no data points in,
    # if they also pass in either ell or theta values to fill it with.
    # They can pass in either but not both.
    if len(sacc_data.data) == 0:
        if (ell_values is None) and (theta_values is None):
            raise ValueError("Supplied an empty sacc file but no ell_values to fill it")
        elif (ell_values is not None) and (theta_values is not None):
            raise ValueError(
                "Supplied an empty sacc file and both theta and ell values to fill it. Just pick one"
            )
        fill_empty_sacc(sacc_data, ell_values=ell_values, theta_values=theta_values)

    # Use the FireCrown machinery to compute the likelihood and as
    # a by-product the theory
    likelihood = build_likelihood(sacc_data, bias)
    loglike = likelihood.compute_loglike(cosmo)

    sacc_theory = sacc_data.copy()

    # Set everything to zero first in case there are BB measurements
    for d in sacc_theory.data:
        d.value = 0.0

    # Fill in the values of the computed theory
    for i, v in zip(likelihood._tx_indices, likelihood.predicted_data_vector):
        sacc_theory.data[i].value = v

    return sacc_theory


def compute_fiducial_bias(b0, sack, cosmo):
    """
    Return an array of bias values following b[i] = b_0 / D(<z_i>)

    Parameters
    ----------
    b0: float

    sack: Sacc

    cosmo: pyccl.Cosmology
    """
    import pyccl as ccl

    print("Computing bias values b(z)")
    # now get bias as a function of redshift for each lens bin
    bias = {}
    for key, tracer in sack.tracers.items():
        if "lens" in key:
            i = int(key.split("_")[1])
            z_eff = (tracer.z * tracer.nz).sum() / tracer.nz.sum()
            a_eff = 1 / (1 + z_eff)
            bias[i] = b0 / ccl.growth_factor(cosmo, a_eff)
    bias = np.array([bias[i] for i in range(len(bias))])
    print("Found bias:", bias)
    return bias


def smooth_sacc_nz(sack):
    """
    Smooth each n(z) in a sacc object, in-place.

    Parameters
    ----------
    sacc: Sacc

    Returns
    -------
    None
    """

    for key, tracer in sack.tracers.items():
        tracer.nz = smooth_nz(tracer.nz)


def smooth_nz(nz):
    return np.convolve(nz, np.exp(-0.5 * np.arange(-4, 5) ** 2) / 2**2, mode="same")


def build_likelihood(sacc_data, bias):
    import sacc
    import pyccl as ccl
    import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
    import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
    from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
    from firecrown.likelihood.gauss_family.gaussian import ConstGaussian

    if isinstance(sacc_data, (str, pathlib.Path)):
        sacc_data = sacc.Sacc.load_fits(sacc_data)
    else:
        sacc_data = sacc_data.copy()

    sacc_data.add_covariance(np.ones(len(sacc_data)))

    # Creating sources, each one maps to a specific section of a SACC file. In
    # this case src0, src1, src2 and src3 describe weak-lensing probes. The
    # sources are saved in a dictionary since they will be used by one or more
    # two-point function.

    sources = {}
    for tracer_name in sacc_data.tracers.keys():
        if tracer_name.startswith("source"):
            tr = wl.WeakLensing(sacc_tracer=tracer_name)
        elif tracer_name.startswith("lens"):
            tr = nc.NumberCounts(sacc_tracer=tracer_name)
            i = int(tracer_name.split("_")[1])
            tr.bias = bias[i]
            tr.mag_bias = 0.0
        else:
            raise ValueError(f"Unknown tracer in sacc file, non-3x2pt: {tracer}")
        sources[tracer_name] = tr

    # Now that we have all sources we can instantiate all the two-point
    # functions. The weak-lensing sources have two "data types", for each one we
    # create a new two-point function.

    stats = {}
    computable_indices = []
    for dt in sacc_data.get_data_types():
        for t1, t2 in sacc_data.get_tracer_combinations(dt):
            try:
                stat = TwoPoint(
                    source0=sources[t1],
                    source1=sources[t2],
                    sacc_data_type=dt,
                )
                stats[f"{stat}_{t1}_{t2}"] = stat
                computable_indices.extend(sacc_data.indices(dt, (t1, t2)))
            except ValueError as err:
                # B modes and things like that can be in the files
                # but are not supported
                if "is not supported" in str(err):
                    print(f"Setting bin {t1}, {t2} for {dt} to zero")
                    continue
                else:
                    raise

    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.
    lk = ConstGaussian(statistics=list(stats.values()))

    # The read likelihood method is called passing the loaded SACC file, the
    # two-point functions will receive the appropriate sections of the SACC
    # file and the sources their respective dndz.
    lk.read(sacc_data)

    # computable_indices is a mask for all the bins that
    # firecrown can calculate. Some of them, like the BB, EB modes, etc.
    # are just zero in the theory, and FireCrown can't predict them right now.
    # so we record this information because in TXPipe's theory prediction
    # we want to set those to zero
    lk._tx_indices = computable_indices

    return lk
