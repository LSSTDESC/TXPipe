import numpy as np

REF_FLUX = 1e23 * 10 ** (48.6/-2.5)

# follow https://pipelines.lsst.io/v/DM-22499/cpp-api/file/_photo_calib_8h.html
def nanojansky_to_mag_ab(flux):
    return -2.5 * np.log10(flux * 1e-9 / REF_FLUX)

def mag_ab_to_nanojansky(mag):
    return 1e9 * REF_FLUX * 10 ** (-0.4 * mag)

def nanojansky_err_to_mag_ab(flux, flux_err):
    return 2.5 / np.log(10) * (flux_err / flux)


def moments_to_shear(Ixx, Iyy, Ixy):
    b = Ixx + Iyy + 2 * np.sqrt(Ixx * Iyy - Ixy**2)
    e1 = (Ixx - Iyy) / b
    e2 = 2 * Ixy / b
    return e1, e2