from scipy.odr import ODR, Model, RealData, unilinear
import numpy as np


def fit_straight_line(x, y, x_err=None, y_err=None, m0=1.0, c0=0.0, nan_error=False, skip_nan=False):
    """
    Use scipy to fit a straight line, with errors or covariances allowed
    on both x and y.

    Parameters
    ----------
    x: array
        x-coordinate
    y: array
        y-coordinate    
    m0: float
        optional, default=1, guess at gradient
    c0: float
        optional, default=1, guess at intercept
    x_err: array/float
        optional, default=None, errors either 1D std. dev., 2D covariance, or scalar constant, or None for unspecified
    y_err: array/float
        optional, default=None, errors either 1D std. dev., 2D covariance, or scalar constant, or None for unspecified


    Returns
    -------
    m: float
        gradient

    c: float
        intercept
    """

    kwargs = {}
    nx = np.ndim(x_err)


    if skip_nan:
        w = np.isfinite(x) & np.isfinite(y)
        x = x[w]
        y = y[w]
    else:
        w = slice(None)
    
    if x_err is None:
        pass
    elif nx == 0:
        kwargs['sx'] = x_err
    elif nx == 1:
        kwargs['sx'] = x_err[w]
    elif nx == 2:
        kwargs['cov_x'] = x_err[w][:,w]
    else:
        raise ValueError("x_sigma_or_cov must be None, scalar, 1D, or 2D")


    ny = np.ndim(y_err)

    if y_err is None:
        pass
    elif ny == 0:
        kwargs['sy'] = y_err
    elif ny == 1:
        kwargs['sy'] = y_err[w]
    elif nx == 2:
        kwargs['cov_y'] = y_err[w][:,w]
    else:
        raise ValueError("x_sigma_or_cov must be None, scalar, 1D, or 2D")
 

    data = RealData(x, y, **kwargs)
    odr = ODR(data, unilinear, beta0=[m0, c0], maxit=200)
    results = odr.run()

    if results.stopreason != ['Sum of squares convergence']:
        if nan_error:
            return np.nan, np.nan, np.zeros((2,2))*np.nan
        else:
            raise RuntimeError('Failed to straight line' + str(results.stopreason))

    m, c = results.beta
    cov = results.cov_beta
    return m, c, cov

