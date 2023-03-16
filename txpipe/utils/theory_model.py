import os
import pathlib

import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
import sacc
import pyccl as ccl
import numpy as np

def build_likelihood(build_parameters):
    """
    This is a generic 3x2pt theory model, but with only
    bias as a systematic.

    """
    sacc_data = build_parameters['sacc_data']
    bias = build_parameters['bias']
    smooth = build_parameters['smooth']
    cosmo = build_parameters['cosmo']

    # items in the build_parameters are supposed to be
    # just str, int, etc, not complicated parameters.
    # this would abuse that slightly. We could always
    # write a temporary file to disc if needed.
    if isinstance(sacc_data, (str, pathlib.Path)):
        sacc_data = sacc.Sacc.load_fits(sacc_data)
    else:
        sacc_data = sacc_data.copy()

    # Add a dummy covariance, always, because the shot noise one
    # may be non-positive definite
    sacc_data.add_covariance(np.ones(len(sacc_data)), overwrite=True)


    # We can optionally smooth the n(z). This helped in Prat et al.
    if smooth:
        smooth_sacc_nz(sacc_data)


    # If someone explains to me how to have fixed linear bias systematic
    # as parameters instead of passing it in like this then that would
    # be better here.
    if isinstance(bias, float):
        bias = compute_fiducial_bias(bias, sacc_data, cosmo)
    elif isinstance(bias, str):
        bias = np.array(bias.split(','), dtype=float)
    elif bias is None:
        bias = np.ones(len(sacc_data.tracers))

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
            i = int(tracer_name.split('_')[1])
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
    tools = ModelingTools()
    tools._tx_indices = computable_indices

    return lk, tools



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


