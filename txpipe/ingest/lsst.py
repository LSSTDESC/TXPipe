from ..utils import nanojansky_err_to_mag_ab, nanojansky_to_mag_ab, moments_to_shear, mag_ab_to_nanojansky
import numpy as np

def process_photometry_data(data):
    cut = data['refExtendedness'] == 1
    cols = {
        'ra': 'coord_ra',
        'dec': 'coord_dec',
        'tract': 'tract',
        'id': 'objectId',
        'extendedness': 'refExtendedness'
    }
    output = {new_name: data[old_name][cut] for new_name, old_name in cols.items()}
    for band in "ugrizy":
        f = data[f"{band}_cModelFlux"][cut]
        f_err = data[f"{band}_cModelFluxErr"][cut]
        output[f'mag_{band}'] = nanojansky_to_mag_ab(f)
        output[f'mag_err_{band}'] = nanojansky_err_to_mag_ab(f, f_err)
        output[f'snr_{band}'] = f / f_err

        # for undetected objects we use a mock mag of 30
        # to choose mag errors
        f_mock = mag_ab_to_nanojansky(30.0)
        undetected = f <= 0
        output[f'mag_{band}'][undetected] = np.inf
        output[f'mag_err_{band}'][undetected] = nanojansky_err_to_mag_ab(f_mock, f_err[undetected])
        output[f'snr_{band}'][undetected] = 0.0
    return output

def process_shear_data(data):
    cut = data['refExtendedness'] == 1
    cols = {
        'ra': 'coord_ra',
        'dec': 'coord_dec',
        'tract': 'tract',
        'id': 'objectId',
        'extendedness': 'refExtendedness'
    }
    output = {new_name: data[old_name][cut] for new_name, old_name in cols.items()}
    for band in "ugrizy":
        f = data[f"{band}_cModelFlux"][cut]
        f_err = data[f"{band}_cModelFluxErr"][cut]

        # The data seems to have some rows where the flux is
        # tiny but unmasked, and the error has overflowed to nan
        # and is masked. We mask these out.
        bad = (~f.mask) & (f_err.mask)
        f[bad] = np.ma.masked

        output[f'mag_{band}'] = nanojansky_to_mag_ab(f)
        output[f'mag_err_{band}'] = nanojansky_err_to_mag_ab(f, f_err)


        if band == "i":
            output['s2n'] = f / f_err
    
    output["g1"] = data['i_hsmShapeRegauss_e1'][cut]
    output["g2"] = data['i_hsmShapeRegauss_e2'][cut]
    output["T"] = data['i_ixx'][cut] + data['i_iyy'][cut]
    output["flags"] = data["i_hsmShapeRegauss_flag"][cut]


    # Fake numbers! These need to be derived from simulation.
    # In this case 
    output["m"] = np.repeat(-1.184445e-01, f.size)
    output["c1"] = np.repeat(-2.316957e-04, f.size)
    output["c2"] = np.repeat(-8.629799e-05, f.size)
    output["sigma_e"] = np.repeat(1.342084e-01, f.size)
    output["weight"] = np.ones_like(f)

    # PSF components
    output["psf_T_mean"] = data['i_ixxPSF'][cut] + data['i_iyyPSF'][cut]
    psf_g1, psf_g2 = moments_to_shear(data['i_ixxPSF'][cut], data['i_iyyPSF'][cut], data['i_ixyPSF'][cut])
    output["psf_g1"] = psf_g1
    output["psf_g2"] = psf_g2

        
        
    return output

