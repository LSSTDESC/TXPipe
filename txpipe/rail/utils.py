import numpy as np


def convert_unseen(data, bands, undetected, unobserved):
    for b in bands:
        m = data[f'mag_{b}_lsst']
        m[np.isnan(m)] = unobserved
        m[np.isinf(m)] = undetected
