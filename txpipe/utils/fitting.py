from scipy.optimize import curve_fit
import numpy as np
import warnings


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
    if x.size == 0:
        print("ERROR: No data for straight line fit. Returning m=0 c=0")
        return 0.0, 0.0, np.array([[1.0, 0.0], [0.0, 1.0]])

    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    try:
        popt, cov = curve_fit(line, x, y, sigma=y_err)
    except RuntimeError:
        print("ERROR: Straight line fit failed. Returning m=0 c=0")
        return 0.0, 0.0, np.array([[1.0, 0.0], [0.0, 1.0]])
    m = popt[0]
    c = popt[1]
    return m, c, cov


def line(slp, vrbl, intcpt):
    return slp * vrbl + intcpt


def calc_chi2(y, err, yfit, v=False):
    """
    Compute chi2 between data and fitted curve

    Parameters
    ----------
    y: array
        y values of data
    err: array (1D or 2D)
        either error bars (if independent data points)
        or covariance matrix
    yfit: array
        fitted values of data
    v: bool
        verbose output

    Returns
    -------
    chi2: float
    """
    if err.shape == (len(y), len(y)):
        # use full covariance
        if v:
            print("cov_mat chi2")
        inv_cov = np.linalg.inv(np.matrix(err))
        chi2 = 0
        for i in range(len(y)):
            for j in range(len(y)):
                chi2 = chi2 + (y[i] - yfit[i]) * inv_cov[i, j] * (y[j] - yfit[j])
        return chi2

    elif err.shape == (len(y),):
        if v:
            print("diagonal chi2")
        return sum(((y - yfit) ** 2.0) / (err**2.0))
    else:
        raise IOError("error in err or cov_mat input shape")
