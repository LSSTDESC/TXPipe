META_VARIANTS = ["00", "1p", "1m", "2p", "2m"]


def metacal_variants(*names):
    return [name + suffix for suffix in ["", "_1p", "_1m", "_2p", "_2m"] for name in names]


def metadetect_variants(*names):
    return [f"{group}/{name}" for group in META_VARIANTS for name in names]


def band_variants(bands, *names, shear_catalog_type="metacal"):
    if shear_catalog_type == "metacal":
        return [
            name + "_" + band + suffix
            for suffix in ["", "_1p", "_1m", "_2p", "_2m"]
            for band in bands
            for name in names
        ]
    elif shear_catalog_type == "metadetect":
        return [
            f"{group}/{name}_{band}" for group in META_VARIANTS for band in bands for name in names
        ]
    else:
        return [name + "_" + band for band in bands for name in names]

