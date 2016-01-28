from sklearn.linear_model import *
import numpy as np
import pylab
from scipy.interpolate import (Rbf,
                               InterpolatedUnivariateSpline,
                               UnivariateSpline)


def save_fig(prefix):
    "Save current figure in extended postscript and PNG formats."
    pylab.savefig('%s.png' % prefix, format='PNG')
    pylab.savefig('%s.eps' % prefix, format='EPS')


def polinomial_kernel(x, degree=4):
    exponents = range(1, degree + 1)
    return [x**exp for exp in exponents]


def model_lightcurve_lasso(df, plot=False, xmin=48823.48, xmax=51663.34,
                           alpha=1E-10):
    y = df['mag']
    local_xmin, local_xmax = min(df.index.values), max(df.index.values)
    x_norm = [(x - xmin)/(xmax - xmin) + 1 for x in df.index.values]
    x = [polinomial_kernel(x, degree=15) for x in x_norm]
    lasso = Ridge(alpha=alpha).fit(x, y)
    x_sample_norm = ([(x - xmin)/(xmax - xmin) + 1
                     for x in np.arange(local_xmin, local_xmax, 2.0)])
    x_sample = [polinomial_kernel(x, degree=15) for x in x_sample_norm]
    y_pred = lasso.predict(x_sample)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x_norm, df['mag'], 'b.')
        plt.plot(x_sample_norm, y_pred, 'r.')
        plt.title("alpha = {0}".format(alpha))
        # plt.xlim(xmin,xmax)
        plt.show()
        plt.close()
    else:
        return y_pred


def model_lightcurve_rbf(df, f='linear', t_w=250, w=100, plot=False):
    x, y = df.index.values.tolist(), df['mag']
    rbf = Rbf(x, y, function=f)
    xi = np.linspace(min(x), min(x) + t_w, w)
    yi = rbf(xi)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x, y, 'b.')
        plt.plot(xi, yi, 'g-')
        plt.show()
        plt.close()
    else:
        return yi, rbf
