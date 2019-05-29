
def metacal_variants(*names):
    return [
        name + suffix
        for suffix in ['', '_1p', '_1m', '_2p', '_2m']
        for name in names
    ]
def metacal_band_variants(*names, bands='riz'):
    return [
        name + "_" + band + suffix
        for suffix in ['', '_1p', '_1m', '_2p', '_2m']
        for band in bands
        for name in names
    ]
