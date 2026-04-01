import numpy as np

REF_FLUX = 1e23 * 10 ** (48.6 / -2.5)


# follow https://pipelines.lsst.io/v/DM-22499/cpp-api/file/_photo_calib_8h.html
def nanojansky_to_mag_ab(flux):
    if isinstance(flux, np.ndarray):
        zero = flux == 0
        out = np.empty_like(flux)
        out[zero] = np.inf
        out[~zero] = -2.5 * np.log10(flux[~zero] * 1e-9 / REF_FLUX)
    elif flux == 0:
        out = np.inf
    
    return out


def mag_ab_to_nanojansky(mag):
    return 1e9 * REF_FLUX * 10 ** (-0.4 * mag)


def nanojansky_err_to_mag_ab(flux, flux_err):
    return 2.5 / np.log(10) * (flux_err / flux)


def moments_to_shear(Ixx, Iyy, Ixy):
    b = Ixx + Iyy + 2 * np.sqrt(Ixx * Iyy - Ixy**2)
    e1 = (Ixx - Iyy) / b
    e2 = 2 * Ixy / b
    return e1, e2

def mag_ab_err_to_nanojansky_err(mag, mag_err):
    flux = mag_ab_to_nanojansky(mag)
    with np.errstate(invalid='ignore'):
        flux_err = flux * np.log(10) / 2.5 * mag_err
    return flux_err