import numpy as np

# follow https://pipelines.lsst.io/v/DM-22499/cpp-api/file/_photo_calib_8h.html
def nanojansky_to_mag_ab(flux):
    ref_flux = 1e23 * 10 ** (48.6/-2.5)
    return -2.5 * np.log10(flux * 1e-9 / ref_flux)

def nanojansky_err_to_mag_ab(flux, flux_err):
    return 2.5 / np.log(10) * (flux_err / flux)


def moments_to_shear(Ixx, Iyy, Ixy):
    b = Ixx + Iyy + 2 * np.sqrt(Ixx * Iyy - Ixy**2)
    e1 = (Ixx - Iyy) / b
    e2 = 2 * Ixy / b
    return e1, e2