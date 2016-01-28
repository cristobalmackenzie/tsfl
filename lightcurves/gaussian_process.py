import infpy.gp
import infpy.gp.kernel_short_names as kernels
from infpy.gp.se_kernel import *
import pylab


# a function to save plots
def save_fig(prefix):
    "Save current figure in extended postscript and PNG formats."
    pylab.savefig('%s.png' % prefix, format='PNG')
    pylab.savefig('%s.eps' % prefix, format='EPS')

"""
df: lightcurve dataframe, mjd in index, mag and err in columns

return: array or list of samples taken at uniform timesteps
"""
def model_lightcurve(df, l=5.0, plot=False):
    X = [[x] for x in df.index.values]
    y = infpy.gp.gaussian_process.gp_zero_mean(df['mag'].tolist())
    # std = 1e-1
    std = 2*df['err'].std()
    K = kernels.SE([l]) + kernels.Noise(sigma=std)
    gp = infpy.gp.GaussianProcess(X, y, K)
    # infpy.gp.gp_learn_hyperparameters(gp)
    x_min, x_max = min(df.index.values), max(df.index.values)
    if plot:
        infpy.gp.gp_1D_predict(gp, num_steps=100, x_min=x_min, x_max=x_max)
    x_support = infpy.gp.gaussian_process.gp_1D_X_range(
            xmin=min(df.index.values),
            xmax=x_min + 2.0*100,
            step=2.0)
    # y_pred = infpy.gp.gaussian_process.gp_sample_from(gp, x_support)
    # return y_pred
    return gp.predict(x_support)
