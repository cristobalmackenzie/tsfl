import pandas as pd
import numpy as np


def normalize(df):
    """
    Recibe un pandas DataFrame con los datos en las filas y retorna un DataFrame
    con las filas normalizadas haciendo (x - mu)/sigma
    """
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    df = df.sub(df_mean, axis=0)
    df = df.div(df_std, axis=0)
    return df


def whiten(df, epsilon=1E-5):
    """
    Recibe una matrix de datos en filas de un pandas DataFrame.
    Se asume que los datos estan normalizados.
    """
    from numpy import dot, sqrt, diag
    from numpy.linalg import eigh

    # covariance matrix
    Xcov = dot(df.T, df) / df.shape[1]

    # eigenvalue decomposition of the covariance matrix
    d, V = eigh(Xcov)

    # an epsilon factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified
    D = diag(1./sqrt(d + epsilon))

    # whitening matrix
    W = dot(dot(V, D), V.T)

    # multiply by the whitening matrix
    X = dot(df, W)

    return X, W


def eliminate_lc_noise(df):
    err_mean = df['err'].mean()
    return df[df['err'] < 3*err_mean]
