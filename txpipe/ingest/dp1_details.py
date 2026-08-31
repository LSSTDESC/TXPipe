# All TRACT INFORMATION SHOULD BE MOVED ELSEWHERE
DP1_COSMOLOGY_FIELDS = [
    "EDFS",
    "ECDFS",
    "LGLF",
]


DP1_TRACTS = {
    # Euclid Deep Field South
    "EDFS": [2393, 2234, 2235, 2394],
    # Extended Chandra Deep Field South
    "ECDFS": [5062, 5063, 5064, 4848, 4849],
    # Low Galactic Latitude Field / Rubin_SV_095_-25
    "LGLF": [5305, 5306, 5525, 5526],
    # Fornax Dwarf Spheroidal Galaxy
    "FDSG": [4016, 4217, 4218, 4017],
    # Low Ecliptic Latitude Field / Rubin_SV_38_7
    "LELF": [10464, 10221, 10222, 10704, 10705, 10463],
    # Seagull Nebula
    "Seagull": [7850, 7849, 7610, 7611],
    # 47 Tuc Globular Cluster
    "47Tuc": [531, 532, 453, 454],
}

DP1_COSMOLOGY_TRACTS = sum([DP1_TRACTS[_field] for _field in DP1_COSMOLOGY_FIELDS], [])
ALL_TRACTS = sum(DP1_TRACTS.values(), [])


# In case useful later:
DP1_FIELD_CENTERS = {
    "47 Tuc Globular Cluster": (6.02, -72.08),
    "Low Ecliptic Latitude Field": (37.86, 6.98),
    "Fornax Dwarf Spheroidal Galaxy": (40.00, -34.45),
    "Extended Chandra Deep Field South": (53.13, -28.10),
    "Euclid Deep Field South": (59.10, -48.73),
    "Low Galactic Latitude Field": (95.00, -25.00),
    "Seagull Nebula": (106.23, -10.51),
}


DP1_SURVEY_PROPERTIES = {
    "deepCoadd_exposure_time_consolidated_map_sum": "Total exposure time accumulated per sky position (second)",
    "deepCoadd_epoch_consolidated_map_min": "Earliest observation epoch (MJD)",
    "deepCoadd_epoch_consolidated_map_max": "Latest observation epoch (MJD)",
    "deepCoadd_epoch_consolidated_map_mean": "Mean observation epoch (MJD)",
    "deepCoadd_psf_size_consolidated_map_weighted_mean": "Weighted mean of PSF characteristic width as computed from the determinant radius (pixel)",
    "deepCoadd_psf_e1_consolidated_map_weighted_mean": "Weighted mean of PSF ellipticity component e1",
    "deepCoadd_psf_e2_consolidated_map_weighted_mean": "Weighted mean of PSF ellipticity component e2",
    "deepCoadd_psf_maglim_consolidated_map_weighted_mean": "Weighted mean of PSF flux 5σ magnitude limit (magAB)",
    "deepCoadd_sky_background_consolidated_map_weighted_mean": "Weighted mean of background light level from the sky (nJy)",
    "deepCoadd_sky_noise_consolidated_map_weighted_mean": "Weighted mean of standard deviation of the sky level (nJy)",
    "deepCoadd_dcr_dra_consolidated_map_weighted_mean": "Weighted mean of DCR-induced astrometric shift in right ascension direction, expressed as a proportionality factor",
    "deepCoadd_dcr_ddec_consolidated_map_weighted_mean": "Weighted mean of DCR-induced astrometric shift in declination direction, expressed as a proportionality factor",
    "deepCoadd_dcr_e1_consolidated_map_weighted_mean": "Weighted mean of DCR-induced change in PSF ellipticity (e1), expressed as a proportionality factor",
    "deepCoadd_dcr_e2_consolidated_map_weighted_mean": "Weighted mean of DCR-induced change in PSF ellipticity (e2), expressed as a proportionality factor",
}
