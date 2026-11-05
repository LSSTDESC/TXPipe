import numpy as np

REF_FLUX = 1e23 * 10 ** (48.6 / -2.5)


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

def mag_ab_to_nanojansky_with_errors(mag, mag_err):
    flux = mag_ab_to_nanojansky(mag)
    with np.errstate(invalid='ignore'):
        flux_err = flux * np.log(10) / 2.5 * mag_err
    return flux, flux_err

def mags_to_snr(mags, bands):
    snr2 = np.zeros_like(mags["mag_" + bands[0]])
    for b in bands:
        mag = mags[f"mag_{b}"]
        mag_err = mags[f"mag_err_{b}"]
        flux, flux_err = mag_ab_to_nanojansky_with_errors(mag, mag_err)
        w = np.where((flux > 0) & np.isfinite(flux_err))
        snr2[w] += (flux[w] / flux_err[w])**2
    return np.sqrt(snr2)

def combine_intrinsic_shear(g1, g2, e1, e2):
    g = g1 + 1j * g2
    ei = e1 + 1j * e2
    e = (ei + g) / (1 + g.conj() * ei)
    e1 = e.real
    e2 = e.imag
    return e1, e2
