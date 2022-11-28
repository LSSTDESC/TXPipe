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



def theory_3x2pt(cosmo, sacc_data, bias=None, smooth=False):
    """
    Use FireCrown to generate the theory predictions for the data
    in a sacc file.

    Linear galaxy bias parameters can optionally be provided as an array.
    If left as None they are set to unity.

    Parameters
    ----------
    cosmo: pyccl.Cosmology
        The cosmology at which to get the theory predictions
    sacc_data: sacc.Sacc or str
        Sacc object or a path to one
    galaxy_bias: float, array, or None
        If a float b_0, use the scheme b = b_0 / D(<z>) for each bin.
        If an array, use the given value for each bin

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

    if smooth:
        sacc_data = smooth_sacc_nz(sacc_data)

    # breakpoint()

    if isinstance(bias, float):
        bias = compute_fiducial_bias(bias, sacc_data, cosmo)
    elif bias is None:
        bias = [1 for _ in sacc_data.tracers]
        print("Using bias = 1")
    else:
        print("Bias = ", bias)

    # Creating sources. This will presumably be done inside
    # firecrown at some point
    sources = {}
    for tracer_name in sacc_data.tracers.keys():
        if tracer_name.startswith("source"):
            tr = wl.WeakLensing(sacc_tracer=tracer_name)
        elif tracer_name.startswith("lens"):
            tr = nc.NumberCounts(sacc_tracer=tracer_name)
            i = int(tracer_name.split('_')[1])
            tr.bias = bias[i]
            tr.mag_bias = 0.0
        else:
            raise ValueError(f"Unknown tracer in sacc file, non-3x2pt: {tracer}")
        sources[tracer_name] = tr

    # Making predictions using the TwoPoint statistic object
    # from firecrown
    theory = {}
    for dt in sacc_data.get_data_types():
        for t1, t2 in sacc_data.get_tracer_combinations(dt):
            try:
                stat = TwoPoint(
                        source0=sources[t1],
                        source1=sources[t2],
                        sacc_data_type=dt,
                )
            except ValueError as err:
                # B modes and things like that can be in the files
                # but are not supported
                if "is not supported" in str(err):
                    print(f"Setting bin {t1}, {t2} for {dt} to zero")
                    n = sacc_data.indices(dt, (t1,t2)).size
                    theory[dt, t1, t2] = np.zeros(n)
                    continue
                else:
                    raise

            # Read statistic information from the sacc file
            stat.read(sacc_data)

            # The first output is the data vector, and
            # we keep the second which is the theory
            _, theory[dt, t1, t2] = stat.compute(cosmo)
        

    # Put the theory values into the new sacc object
    # We really need a set_mean on sacc objects to do this.
    sacc_theory = sacc_data.copy()
    for dt in sacc_theory.get_data_types():
        for t1, t2 in sacc_theory.get_tracer_combinations(dt):
            th = theory[dt, t1, t2]
            index = sacc_theory.indices(dt, (t1,t2))
            assert len(index) == len(th)
            for i,j in enumerate(index):
                sacc_theory.data[j].value = th[i]

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
            i = int(key.split('_')[1])
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
