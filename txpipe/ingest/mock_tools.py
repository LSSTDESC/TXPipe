import numpy as np
from ..utils.conversion import mag_ab_to_nanojansky, nanojansky_to_mag_ab, nanojansky_err_to_mag_ab, mag_ab_err_to_nanojansky_err



# These are quick, rough, and global numbers JZ measured on DES Y6
# metadetect catalogs to use as defaults for response values
# The value d(parameter)/d(g_i) has the key (parameter, i)
desy6_responses = {
    ("g1", 1): 0.7522786345399349,
    ("g1", 2): -7.221603811297746e-05,
    ("g2", 1): 0.000695986991087691,
    ("g2", 2): 0.753689800584438,
    ("s2n", 1): 0.4495460648463734,
    ("s2n", 2): -0.2446177233672131,
    ("flux_g", 1): 2.200867381810667,
    ("flux_g", 2): -0.7594371207147788,
    ("flux_r", 1): 6.4665622482152685,
    ("flux_r", 2): -3.319059863883922,
    ("flux_i", 1): 9.973907233552382,
    ("flux_i", 2): -5.844535374410498,
    ("flux_z", 1): 13.01874158668852,
    ("flux_z", 2): -7.8327478646315285,
    ("T", 1): 0.0002450368855461127,
    ("T", 2): -0.0005327801992860426,
}


# These are means and std devs in the i band from running this notebook:
# https://github.com/LSSTDESC/A360_DP1/blob/main/ACO360_PSF_properties_RSP.ipynb
dp1_psf_stats = {
    # trace size in arcsec^2
    "T": (0.424, 0.025),  
    "e1": (-0.065, 0.0325),
    "e2": (-0.00584, 0.02798), 
}

unit_responses = {
    ("g1", 1): 1.0,
    ("g1", 2): 0.0,
    ("g2", 1): 0.0,
    ("g2", 2): 1.0,
    ("s2n", 1): 0.0,
    ("s2n", 2): 0.0,
    ("flux_g", 1): 0.0,
    ("flux_g", 2): 0.0,
    ("flux_r", 1): 0.0,
    ("flux_r", 2): 0.0,
    ("flux_i", 1): 0.0,
    ("flux_i", 2): 0.0,
    ("flux_z", 1): 0.0,
    ("flux_z", 2): 0.0,
    ("T", 1): 0.0,
    ("T", 2): 0.0,
}



def add_lsst_like_noise(data, random_state, year=1, **config):
    from photerr import LsstErrorModel
    import pandas as pd

    # LsstErrorModel is the newer of the photometry models
    # and should be more representative of real data.

    # Find all bands available in the data
    bands = [b for b in "ugrizy" if f"mag_{b}" in data]

    model = LsstErrorModel(
        nYrObs=year,
        **config
    )

    df = pd.DataFrame({b: data[f"mag_{b}"] for b in bands})
    noisy_data = model(df, random_state=random_state)

    # Keep all the input columns as expected.

    for b in bands:
        mag = np.array(noisy_data[b].values)
        mag_err = np.array(noisy_data[f"{b}_err"].values)
        flux = mag_ab_to_nanojansky(mag)
        flux_err = mag_ab_err_to_nanojansky_err(mag, mag_err)
        snr = flux / flux_err
        data[f"mag_{b}"] = mag
        data[f"mag_err_{b}"] = mag_err
        data[f"snr_{b}"] = snr
        data[f"s2n_{b}"] = snr




def make_metadetect_catalog(data, response_type, delta_gamma, bands, rng, snr_cut=5, T_ratio_cut=0.5):
    output = {}

    # These parameters have response factors defined for them.
    params = ["g1", "g2", "T"]

    # For these parameters we do not apply any response correction,
    # they have the same value in the metadetect variants.
    # Note that this will not be the case in real life - different
    # sets of galaxies will be identified in each metadetect variant,
    # and locations will change.
    non_response_params = ["ra", "dec", "id", "redshift_true"]

    if response_type == "desy6":
        responses = desy6_responses
    elif response_type == "unit":
        responses = unit_responses
    elif response_type == "none":
        responses = None
    else:
        raise ValueError(f"Unknown response type {response_type}")
    
    # If there is no response then we are not doing metacal at all.
    # In that case we just want to 
    if responses is None:
        for old_name, new_name in params + non_response_params:
            output[new_name] = data[old_name]
        psf_T, psf_e1, psf_e2 = make_mock_psf(output["00/ra"].size, rng)
        output["psf_T"] = psf_T
        output["psf_g1"] = psf_e1
        output["psf_g2"] = psf_e2


    for param in params:
        output[f"00/{param}"] = data[param]
        output[f"1p/{param}"] = data[param] + responses[param, 1] * delta_gamma / 2
        output[f"1m/{param}"] = data[param] - responses[param, 1] * delta_gamma / 2
        output[f"2p/{param}"] = data[param] + responses[param, 2] * delta_gamma / 2
        output[f"2m/{param}"] = data[param] - responses[param, 2] * delta_gamma / 2

    for param in non_response_params:
        output[f"00/{param}"] = data[param]
        output[f"1p/{param}"] = data[param]
        output[f"1m/{param}"] = data[param]
        output[f"2p/{param}"] = data[param]
        output[f"2m/{param}"] = data[param]

    s2n_total = 0
    s2n_total_1p = 0
    s2n_total_1m = 0
    s2n_total_2p = 0
    s2n_total_2m = 0

    dg2 = delta_gamma / 2

    for b in bands:
        mag = data["mag_" + b]
        mag_err = data["mag_err_" + b]
        flux = mag_ab_to_nanojansky(mag)
        flux_err = mag_ab_err_to_nanojansky_err(mag, mag_err)

        s2n = flux / flux_err
        s2n_1p = s2n + responses["s2n", 1] * dg2
        s2n_1m = s2n - responses["s2n", 1] * dg2
        s2n_2p = s2n + responses["s2n", 2] * dg2
        s2n_2m = s2n - responses["s2n", 2] * dg2

        s2n_total += s2n ** 2
        s2n_total_1p += s2n_1p ** 2
        s2n_total_1m += s2n_1m ** 2
        s2n_total_2p += s2n_2p ** 2
        s2n_total_2m += s2n_2m ** 2

        flux_1p = flux + responses[(f"flux_{b}", 1)] * dg2
        flux_1m = flux - responses[(f"flux_{b}", 1)] * dg2
        flux_2p = flux + responses[(f"flux_{b}", 2)] * dg2
        flux_2m = flux - responses[(f"flux_{b}", 2)] * dg2
        # there are some zero fluxes for non-detections. We silence the warning
        with np.errstate(divide = 'ignore'):
            output[f"00/mag_{b}"] = nanojansky_to_mag_ab(flux)
            output[f"1p/mag_{b}"] = nanojansky_to_mag_ab(flux_1p)
            output[f"1m/mag_{b}"] = nanojansky_to_mag_ab(flux_1m)
            output[f"2p/mag_{b}"] = nanojansky_to_mag_ab(flux_2p)
            output[f"2m/mag_{b}"] = nanojansky_to_mag_ab(flux_2m)

            

    output["00/s2n"] = np.sqrt(s2n_total)
    output["1p/s2n"] = np.sqrt(s2n_total_1p)
    output["1m/s2n"] = np.sqrt(s2n_total_1m)
    output["2p/s2n"] = np.sqrt(s2n_total_2p)
    output["2m/s2n"] = np.sqrt(s2n_total_2m)

    # Make fake PSF choices. Give the same PSF to each variant for simplicity.
    psf_T, psf_e1, psf_e2 = make_mock_psf(output["00/ra"].size, rng)
    for variant in ["00", "1p", "1m", "2p", "2m"]:
        output[f"{variant}/psf_T"] = psf_T
        output[f"{variant}/psf_g1"] = psf_e1
        output[f"{variant}/psf_g2"] = psf_e2

    # Now we do the cuts on each variant separately
    for variant in ["00", "1p", "1m", "2p", "2m"]:
        T = output[f"{variant}/T"]
        psf_T = output[f"{variant}/psf_T"]
        snr = output[f"{variant}/s2n"]
        mask = compute_cuts(T, psf_T, snr)
        for key in output:
            if key.startswith(f"{variant}/"):
                output[key] = output[key][mask]

    return output


def compute_cuts(T, psf_T, snr, snr_cut=5, T_ratio_cut=0.5):
    """Apply basic cuts to a metadetect catalog dictionary."""
    T_ratio = T / psf_T
    mask = (snr > snr_cut) & (T_ratio > T_ratio_cut)
    return mask

def make_mock_psf(n, rng):
    T = rng.normal(dp1_psf_stats["T"][0], dp1_psf_stats["T"][1], n)
    e1 = rng.normal(dp1_psf_stats["e1"][0], dp1_psf_stats["e1"][1], n)
    e2 = rng.normal(dp1_psf_stats["e2"][0], dp1_psf_stats["e2"][1], n)
    return T, e1, e2


def generate_mock_metacal_mag_responses(bands, nobj):
    nband = len(bands)
    mu = np.zeros(nband)  # seems approx mean of response across bands, from HSC tract
    rho = 0.25  #  approx correlation between response in bands, from HSC tract
    sigma2 = 1.7**2  # approx variance of response, from HSC tract
    covmat = np.full((nband, nband), rho * sigma2)
    np.fill_diagonal(covmat, sigma2)
    mag_responses = np.random.multivariate_normal(mu, covmat, nobj).T
    return mag_responses
