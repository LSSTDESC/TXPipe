import numpy as np
import warnings

# same convention as elsewhere
SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2


def theory_3x2pt(cosmology_file, tracers, nbin_source, nbin_lens, fourier=True):
    """Compute the 3x2pt theory, for example for a fiducial cosmology

    Parameters
    ----------
    cosmology_file: str
        name of YAML file

    tracers: dict{str:obj}
        dict of objects (e.g. sacc tracers) containing z, nz.
        keys are source_0, source_1, ..., lens_0, lens_1, ...
    
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

    ell_max = 2000 if fourier else 100_000
    n_ell = 100 if fourier else 200
    ell = np.logspace(1, np.log10(ell_max), n_ell).astype(int)
    ell = np.unique(ell)


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
            theory_cl [(j,i,k)] = theory_cl [(i,j,k)]
            

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
            theory_cl [(j,i,k)] = theory_cl [(i,j,k)]

    if fourier:
        return theory_cl

    theta_min = 1.0 / 60
    theta_max = 3.0
    theta = np.logspace(np.log(theta_min), np.log(theta_max), 200)

    theory_xi = {'theta': theta*60} # arcmin
    for key, val in theory_cl.items():
        if key == 'ell':
            continue
        i,j,k = key
        corr_type = {SHEAR_SHEAR: 'L+', POS_POS: 'GG', SHEAR_POS: 'GL'}[k]
        xi = ccl.correlation(cosmo, ell, val, theta, corr_type=corr_type)
        if k == SHEAR_SHEAR:
            xim = ccl.correlation(cosmo, ell, val, theta, corr_type='L-')
            theory_xi[(i,j,k)] = [xi,xim]
        else:
            theory_xi[(i,j,k)] = xi

    return theory_cl, theory_xi