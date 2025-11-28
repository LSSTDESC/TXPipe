"""
This function and script are from the COSMOS2020 website and process
the photometry to apply aperture-to-total corrections and Milky Way
attenuation corrections.

See https://cosmos2020.calet.org/catalogues/ readme for details.
"""

import numpy as np
import sys
from astropy.io import fits



def get_total_flux_cosmos2020(
    data,
    bands,
):
    out = {}

    # The catalog has a pre-computed offset to go from aperture to total
    offset = data["total_off2"]
    
    for band in bands:
        mag_column = band + "_MAG_APER2"
        mag_err_column = band + "_MAGERR_APER2"

        # Just output as "u", "g", etc instead of "CFHT_u", etc
        short_band = band.split("_")[1]

        out[short_band] = data[mag_column] + offset
        # The error on the magnitude does not change with the offset
        # to the total magnitude.
        out["mag_err_" + short_band] = data[mag_err_column]
    return out
