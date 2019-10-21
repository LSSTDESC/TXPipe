
def metacal_variants(*names):
    return [
        name + suffix
        for suffix in ['', '_1p', '_1m', '_2p', '_2m']
        for name in names
    ]
def metacal_band_variants(bands, *names):
    return [
        name + "_" + band + suffix
        for suffix in ['', '_1p', '_1m', '_2p', '_2m']
        for band in bands
        for name in names
    ]

def apply_response(R,g1,g2):
    R11 = R[0][0]
    R22  = R[1][1]
    return g1/R11, g2/R22