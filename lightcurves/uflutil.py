import preprocess as pp
import numpy as np
import pandas as pd
import lightcurveutil as lcu
import lc_models as lcm
from random import randint, random
from sklearn.decomposition import SparseCoder


def sample_from_files(paths, num_samples, t_w=250, w=100, dataset='macho'):
    """
    paths: list with file paths from which to sample
    num_samples: number of samples to take

    return: lcs = list with each sample in a pandas DataFrame
    """
    # lightcurves that are opened once are stored in a dictionary to avoid
    # opening and closing the same file many times
    n = len(paths)
    print "Sampling from {0} file paths".format(n)
    samples_taken = 0
    samples = []
    modeled_samples = []
    times = []
    samples_per_lc = (num_samples/n) + 1
    print "Samples per lc: {0}".format(samples_per_lc)
    for i in range(n):
        # sample the lc path
        path = paths[i]
        lc = lcu.open_lightcurve(path, dataset=dataset)
        # lc = pp.eliminate_lc_noise(lc)
        try:
            yi, rbf = lcm.model_lightcurve_rbf(
                lc, f='linear', t_w=t_w, w=w, plot=False)
            j = 0
            num_errors = 0
            while j < samples_per_lc:
                # sample a piece of the lightcurve dataframe, of time length t_w
                t_min, t_max = min(lc.index), max(lc.index)
                t_sample = t_min + (t_max - t_min - t_w)*random()
                lc_s = lc[np.logical_and(lc.index > t_sample,
                                         lc.index <= t_sample + t_w)]
                if len(lc) > 20:
                    rbf_sample_times = np.linspace(t_sample, t_sample + t_w, w)
                    rbf_values = rbf(rbf_sample_times)
                    modeled_samples.append(rbf_values)
                    samples.append(lc_s)
                    times.append(t_sample)
                    samples_taken += 1
                    print "samples_taken: {0}".format(samples_taken)
                    j += 1
                else:
                    print "sampling error"
                    num_errors += 1
                if num_errors > 10:
                    print "Couldn't sample from lc with path: {0}".format(path)
                    break
        except Exception as e:
            print e
            print "lc index {0} raised an error.".format(i)
    return samples, modeled_samples, times


def sample_from_df(df, num_samples, w=100):
    n_ul, m_ul = df.shape
    k = 0
    samples = {}
    i = 0
    while k < num_samples:
        i = k % n_ul
        j = randint(0, m_ul - w - 1)
        # i, j = randint(0, n_ul - 1), randint(0, m_ul - w - 1)
        sample = df.iloc[i,j:j+w]
        samples[k] = sample.tolist()
        k += 1
    X = pd.DataFrame(samples)
    return X.T


def encode_kmeans_triangle(df, km, split=False, alpha=1):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    mean = df.mean(axis=1) # la media de cada columna
    df = df.div(-1)
    df = df.add(alpha*mean, axis=0) # Dejar todos los valores en Mu - X
    if split:
        df1 = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
        df2 = df.apply(lambda x: np.maximum(0,-x)) # Dejar todos los positivos en 0
        df = df1.merge(df2, left_index=True, right_index=True)
    else:
        df = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
    return df


def encode_kmeans_triangleok(df, km, split=False, alpha=1):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    mean = df.mean(axis=0) # la media de cada fila
    df = df.div(-1)
    df = df.add(alpha*mean, axis=1) # Dejar todos los valores en Mu - X
    if split:
        df1 = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
        df2 = df.apply(lambda x: np.maximum(0,-x)) # Dejar todos los positivos en 0
        df = df1.merge(df2, left_index=True, right_index=True)
    else:
        df = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
    return df


def encode_kmeans_hard(df, km):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    df = df.apply(lambda x: x == min(x), axis=1) # True solo donde en el indice del cluster mas cercano
    df = df.applymap(int) # Cambiar todo a int
    return df


def encode_kmeans_soft_threshold(df, km, alpha=0.5):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    df = df.apply(lambda x: np.maximum(0, x - alpha)) # Poner el 0 los que quedan negativos
    return df


def encode_kmeans_sparsecode(df, km, algo='lasso_cd', alpha=1, split=False):
    centroids = km.cluster_centers_
    D = [centroids[i]/np.linalg.norm(centroids[i]) for i in range(len(centroids))]
    D = np.array(D)
    sc = SparseCoder(D, transform_algorithm=algo, transform_alpha=alpha, split_sign=split)
    return pd.DataFrame(sc.transform(df))


def encode_lightcurve(df, km, method='triangle', alpha=1, split=False):
    if method == 'triangle':
        return encode_kmeans_triangle(df, km, split=split, alpha=alpha)
    if method == 'triangleok':
        return encode_kmeans_triangleok(df, km, split=split, alpha=alpha)
    if method == 'sparse':
        return encode_kmeans_sparsecode(df, km, alpha=alpha, split=split)
    if method == 'threshold':
        return encode_kmeans_soft_threshold(df, km, alpha)


def encode_lightcurve_twed(cluster_lcs, cluster_times, patches_lcs,
                           patches_times, twed_func, lam=.5, nu=1e-5, alpha=1,
                           split=True):
    num_patches = len(patches_lcs)
    num_centroids = len(cluster_lcs)
    D = np.zeros((num_patches, num_centroids))

    for i in xrange(num_patches):
        for j in xrange(num_centroids):
            D[i, j] = twed_func(patches_lcs[i], patches_times[i],
                                cluster_lcs[j], cluster_times[j],
                                lam=lam, nu=nu)

    # Transformado al espacio de cluster distance
    df = pd.DataFrame(D)
    mean = df.mean(axis=1)  # la media de cada columna
    df = df.div(-1)
    df = df.add(alpha*mean, axis=0)  # Dejar todos los valores en Mu - X
    if split:
        # Dejar todos los negativos en 0
        df1 = df.apply(lambda x: np.maximum(0, x))
        # Dejar todos los positivos en 0
        df2 = df.apply(lambda x: np.maximum(0, -x))
        df = df1.merge(df2, left_index=True, right_index=True)
    else:
        # Dejar todos los negativos en 0
        df = df.apply(lambda x: np.maximum(0, x))
    return df


def max_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].max(axis=0).tolist()
    return pooled_datum


def mean_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].mean(axis=0).tolist()
    return pooled_datum


def median_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].median(axis=0).tolist()
    return pooled_datum


def sum_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].sum(axis=0).tolist()
    return pooled_datum


def pool_lightcurve(df, num_pool=4, method='mean'):
    if method == 'max':
        return max_pool(df, num_pool)
    if method == 'mean':
        return mean_pool(df, num_pool)
    if method == 'median':
        return median_pool(df, num_pool)
    if method == 'sum':
        return sum_pool(df, num_pool)
