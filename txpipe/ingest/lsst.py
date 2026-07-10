from ..utils import nanojansky_err_to_mag_ab, nanojansky_to_mag_ab, moments_to_shear, mag_ab_to_nanojansky
import numpy as np
import h5py
from ..shear_calibration.names import META_VARIANTS

def process_photometry_data(data):
    cut = data["refExtendedness"] == 1
    cols = {"ra": "coord_ra", "dec": "coord_dec", "tract": "tract", "id": "objectId", "extendedness": "refExtendedness"}
    output = {new_name: data[old_name][cut] for new_name, old_name in cols.items()}
    for band in "ugrizy":
        f = data[f"{band}_cModelFlux"][cut]
        f_err = data[f"{band}_cModelFluxErr"][cut]
        output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
        output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)
        output[f"snr_{band}"] = f / f_err

        # for undetected objects we use a mock mag of 30
        # to choose mag errors
        f_mock = mag_ab_to_nanojansky(30.0)
        undetected = f <= 0
        output[f"mag_{band}"][undetected] = np.inf
        output[f"mag_err_{band}"][undetected] = nanojansky_err_to_mag_ab(f_mock, f_err[undetected])
        output[f"snr_{band}"][undetected] = 0.0

        # mask out object that have nan errors, we have
        # no way of dealing with these
        err_is_nan = np.isnan(output[f"mag_err_{band}"])
        output[f"mag_{band}"][err_is_nan] = np.nan

    output["flags"] = data["deblend_skipped"][cut] | data["deblend_failed"][cut] | (output["extendedness"] != 1)

    return output


def process_shear_data(data):
    cut = data["refExtendedness"] == 1
    cols = {"ra": "coord_ra", "dec": "coord_dec", "tract": "tract", "id": "objectId", "extendedness": "refExtendedness"}
    output = {new_name: data[old_name][cut] for new_name, old_name in cols.items()}
    for band in "ugrizy":
        f = data[f"{band}_cModelFlux"][cut]
        f_err = data[f"{band}_cModelFluxErr"][cut]
        output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
        output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)

        if band == "i":
            output["s2n"] = f / f_err

    output["g1"] = data["i_hsmShapeRegauss_e1"][cut]
    output["g2"] = data["i_hsmShapeRegauss_e2"][cut]
    output["T"] = data["i_ixx"][cut] + data["i_iyy"][cut]
    output["flags"] = data["i_hsmShapeRegauss_flag"][cut]

    # Fake numbers! These need to be derived from simulation.
    # In this case
    output["m"] = np.repeat(-1.184445e-01, f.size)
    output["c1"] = np.repeat(-2.316957e-04, f.size)
    output["c2"] = np.repeat(-8.629799e-05, f.size)
    output["sigma_e"] = np.repeat(1.342084e-01, f.size)
    output["weight"] = np.ones_like(f)

    # PSF components
    output["psf_T_mean"] = data["i_ixxPSF"][cut] + data["i_iyyPSF"][cut]
    psf_g1, psf_g2 = moments_to_shear(data["i_ixxPSF"][cut], data["i_iyyPSF"][cut], data["i_ixyPSF"][cut])
    output["psf_g1"] = psf_g1
    output["psf_g2"] = psf_g2

    return output


def process_metadetect_data(data, flag):
    output = {}
    for variant in META_VARIANTS:
        var_data = data[data["metaStep"] == variant]
        var_data = sanitize(var_data)

        var_output = {
            "ra": var_data["ra"],
            "dec": var_data["dec"],
            "id": var_data["shearObjectId"],
            "object_mask_fraction": var_data["mfrac"],
            #"n_epoch": var_data["nEpochCell"],
            "g1": var_data["gauss_g1"],
            "g2": var_data["gauss_g2"],
            "g1_err": var_data["gauss_g1_g1_Cov"],
            "g2_err": var_data["gauss_g2_g2_Cov"],
            "g_cross": var_data["gauss_g1_g2_Cov"],
            "T": var_data["gauss_T"],
            "s2n": var_data["gauss_snr"],
            "T_err": var_data["gauss_TErr"],
            "psf_g1_original": var_data["psfOriginal_g1"],
            "psf_g2_original": var_data["psfOriginal_g2"],
            "psf_T_mean_original": var_data["psfOriginal_T"],
            "psf_g1": var_data["gauss_psfReconvolved_g1"],
            "psf_g2": var_data["gauss_psfReconvolved_g2"],
            "psf_T_mean": var_data["gauss_psfReconvolved_T"],
            "flags": combined_flag(var_data, flag),
            "weight": 1 / (0.5 * (var_data["gauss_g1_g1_Cov"] + var_data["gauss_g2_g2_Cov"])),
            "gauss_flags": var_data["gauss_flags"],
            "pgauss_flags": var_data["pgauss_flags"],
            "gauss_shape_flags": var_data["gauss_shape_flags"],
            "is_primary": var_data["is_primary"],
            "gauss_object_flags": var_data["gauss_object_flags"],
            "pgauss_object_flags": var_data["pgauss_object_flags"],
            "psfOriginal_flags": var_data["psfOriginal_flags"],
            "gauss_psfReconvolved_flags": var_data["gauss_psfReconvolved_flags"],
        }
        for band in "griz": # For DP2, we only expect 4 bands
            f = var_data[f"{band}_pgaussFlux"]
            f_err = var_data[f"{band}_pgaussFluxErr"]
            var_output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
            var_output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)
            var_output[f"{band}_gaussFlux_flags"] = var_data[f"{band}_gaussFlux_flags"]
            var_output[f"{band}_pgaussFlux_flags"] = var_data[f"{band}_pgaussFlux_flags"]
        output[f"{variant}"] = var_output

    return output

def sanitize(data):
    """
    Convert unicode arrays into types that h5py can save
    """
    # convert unicode to strings
    if data.dtype.kind == "U":
        data = data.astype("S")
    # convert dates to integers
    elif data.dtype.kind == "M":
        data = data.astype(h5py.opaque_dtype(data.dtype))

    return data


def combined_flag(data, flag):
    """
    generate a combined flag for the metadetect catalog,
    this could also become initial cut if we want it to.
    """
    flag = np.ones(len(data), dtype=bool)
    flag &= data["gauss_object_flags"] == 0
    flag &= data["pgauss_object_flags"] == 0
    flag &= data["is_primary"]
    flag &= data["psfOriginal_flags"] == 0
    flag &= data["gauss_psfReconvolved_flags"] == 0
    flag &= data["gauss_shape_flags"] == 0
    flag &= data["gauss_flags"] == 0
    flag &= data["pgauss_flags"] == 0
    if flag:
        for band in "griz":
            flag &= data[f"{band}_gaussFlux_flags"] == 0
            flag &= data[f"{band}_pgaussFlux_flags"] == 0
    return (~flag).astype(int)
