import numpy as np
import warnings

# same convention as elsewhere
SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2


def theory_3x2pt(cosmology_file, tracers, ell_max, nbin_source, nbin_lens):
    """Compute the 3x2pt theory, for example for a fiducial cosmology

    Parameters
    ----------
    cosmology_file: str
        name of YAML file

    tracers: dict{str:obj}
        dict of objects (e.g. sacc tracers) containing z, nz.
        keys are source_0, source_1, ..., lens_0, lens_1, ...
    
    ell_max: int
        max ell to calculate

    nbin_source: int
        number of source bins

    nbin_lens: int
        number of lens bins

    Returns
    -------
    theory_cl: dict{str:array}
        theory c_ell for each pair (i,j,k) where k is one of
        SHEAR_SHEAR = 0, SHEAR_POS = 1, POS_POS = 2

    """
    import pyccl as ccl

    cosmo = ccl.Cosmology.read_yaml(cosmology_file)

    # We will need the theory C_ell in a continuum up until
    # the full ell_max, because we will need a weighted sum
    # over the values
    ell_max = ell_max
    #f_d[0].fl.lmax
    ell = np.arange(ell_max+1, dtype=int)

    # Convert from SACC tracers (which just store N(z))
    # to CCL tracers (which also have cosmology info in them).
    CTracers = {}

    # Lensing tracers - need to think a little more about
    # the fiducial intrinsic alignment here
    for i in range(nbin_source):
        x = tracers[f'source_{i}']
        tag = ('S', i)
        CTracers[tag] = ccl.WeakLensingTracer(cosmo, dndz=(x.z, x.nz))
    # Position tracers - even more important to think about fiducial biases
    # here - these will be very very wrong otherwise!
    # Important enough that I'll put in a warning.
    warnings.warn("Not using galaxy bias in fiducial theory density spectra")

    for i in range(nbin_lens):
        x = tracers[f'lens_{i}']
        tag = ('P', i) 
        b = np.ones_like(x.z)
        CTracers[tag] = ccl.NumberCountsTracer(cosmo, dndz=(x.z, x.nz),
                                    has_rsd=False, bias=(x.z,b))

    # Use CCL to actually calculate the C_ell values for the different cases
    theory_cl = {}
    theory_cl['ell'] = ell
    k = SHEAR_SHEAR
    for i in range(nbin_source):
        for j in range(i+1):
            Ti = CTracers[('S',i)]
            Tj = CTracers[('S',j)]
            # The full theory C_ell over the range 0..ellmax
            theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)
            theory_cl [(j,i,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)
            

    # The same for the galaxy galaxy-lensing cross-correlation
    k = SHEAR_POS
    for i in range(nbin_source):
        for j in range(nbin_lens):
            Ti = CTracers[('S',i)]
            Tj = CTracers[('P',j)]
            theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)

    # And finally for the density correlations
    k = POS_POS
    for i in range(nbin_lens):
        for j in range(i+1):
            Ti = CTracers[('P',i)]
            Tj = CTracers[('P',j)]
            theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)
            theory_cl [(j,i,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)

    return theory_cl