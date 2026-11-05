"""
A firecrown likelihood for the DES Y3 MagLim 3x2pt analysis, in real space.

It is written for the sacc file made by prepare_data.py, with TXPipe tracer
names (source_0..3, lens_0..3), and follows the DES Y3 model as closely as
firecrown allows:

 - weak lensing: NLA intrinsic alignments with redshift dependence,
   multiplicative shear bias and photo-z shifts per source bin;
 - number counts: linear galaxy bias, photo-z shift and stretch, and
   constant magnification bias per lens bin;
 - non-linear matter power spectrum from CCL (Halofit), and the
   Gaussian likelihood with the DES Y3 covariance.

It does not include the shear-ratio likelihood or point-mass marginalisation
used in the DES Y3 analysis. DES Y3 used TATT intrinsic alignments as its
fiducial model and NLA as an alternative; we use NLA because with TATT the
small-scale shear C_ell can change sign at high ell, which makes the CCL
Hankel transform to xi_pm fail for some cosmologies.

Firecrown calls build_likelihood with a NamedParameters object whose
"sacc_data" entry is either a file name or a sacc.Sacc object.
"""
import pathlib

import sacc

import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood import TwoPoint, ConstGaussian, NamedParameters
from firecrown.modeling_tools import ModelingTools, CCLFactory


def build_likelihood(build_parameters):
    if isinstance(build_parameters, NamedParameters):
        build_parameters = build_parameters.data
    sacc_data = build_parameters["sacc_data"]
    if isinstance(sacc_data, (str, pathlib.Path)):
        sacc_data = sacc.Sacc.load_fits(sacc_data)

    # One NLA model, with redshift dependence, shared by all source bins
    ia_systematic = wl.LinearAlignmentSystematic()

    sources = {}
    for tracer_name in sacc_data.tracers:
        if tracer_name.startswith("source"):
            systematics = [
                ia_systematic,
                wl.MultiplicativeShearBias(sacc_tracer=tracer_name),
                wl.PhotoZShift(sacc_tracer=tracer_name),
            ]
            sources[tracer_name] = wl.WeakLensing(sacc_tracer=tracer_name, systematics=systematics)
        elif tracer_name.startswith("lens"):
            systematics = [
                nc.PhotoZShiftandStretch(sacc_tracer=tracer_name),
                nc.ConstantMagnificationBiasSystematic(sacc_tracer=tracer_name),
            ]
            sources[tracer_name] = nc.NumberCounts(sacc_tracer=tracer_name, systematics=systematics)
        else:
            raise ValueError(f"Unexpected tracer {tracer_name} in DES Y3 3x2pt sacc file")

    # One two-point function per data type and tracer pair in the file,
    # so the theory vector follows the order of the sacc data.
    statistics = [
        TwoPoint(source0=sources[t1], source1=sources[t2], sacc_data_type=data_type)
        for data_type in sacc_data.get_data_types()
        for t1, t2 in sacc_data.get_tracer_combinations(data_type)
    ]

    likelihood = ConstGaussian(statistics=statistics)
    likelihood.read(sacc_data)

    tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True))

    return likelihood, tools
