from ..utils import nanojansky_err_to_mag_ab, nanojansky_to_mag_ab, moments_to_shear, mag_ab_to_nanojansky
import numpy as np


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


def process_metadetect_data(data):
    output = {}
    for variant in ["ns", "1p", "1m", "2p", "2m"]:
        var_data = data[data["metaStep"] == variant]
        var_data = sanitize(var_data)

        var_output = {
            "ra": var_data["ra"],
            "dec": var_data["dec"],
            "id": var_data["shearObjectId"],
            "metaStep": var_data["metaStep"].astype("S"), #might not be needed
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
            "psf_g1": var_data["psfOriginal_e1"],
            "psf_g2": var_data["psfOriginal_e2"],
            "mcal_psf_g1": var_data["gauss_psfReconvolved_g1"],
            "mcal_psf_g2": var_data["gauss_psfReconvolved_g1"],
            "mcal_psf_T_mean": var_data["gauss_psfReconvolved_T"],
            "flags": var_data["gauss_shape_flags"], # TO BE ADDRESSED!
        }
        for band in "gri": # For DP2, we only expect 4 bands
            f = var_data[f"{band}_pgaussFlux"]
            f_err = var_data[f"{band}_pgaussFluxErr"]
            var_output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
            var_output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)
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
        data = data.astype(int)

    return data