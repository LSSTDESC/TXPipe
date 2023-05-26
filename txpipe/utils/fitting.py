
from scipy.optimize import curve_fit
import numpy as np


def fit_straight_line(x, y, y_err=None):
    """
    Use scipy to fit a straight line, with errors bars in y.

    Parameters
    ----------
    x: array
        x-coordinate
    y: array
        y-coordinate
    y_err: array/float
        optional, default=None, errors are 1D std. dev.


    Returns
    -------
    m: float
        gradient

    c: float
        intercept
    """

    popt, cov = curve_fit(line, x, y, sigma=y_err)
    m = popt[0]
    c = popt[1]
    return m,c, cov

def line(slp,vrbl,intcpt):
    return slp * vrbl + intcpt
    
