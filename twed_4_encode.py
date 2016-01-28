import argparse
import numpy as np
import pandas as pd
from twed import twed
import lightcurves.preprocess as pp
import lightcurves.lightcurveutil as lcu
import lightcurves.uflutil as uflul
import os
import sys
import pickle
import datetime
import time

print ' '.join(sys.argv)

parser = argparse.ArgumentParser(
    description='Encode a lightcurve training set.')
parser.add_argument('--training_lcs_file', required=True, type=str)
parser.add_argument('--clustering_file', required=True, type=str)
parser.add_argument('--dataset', default='macho',
                    choices=['macho', 'ogle', 'eros'])
parser.add_argument('--time_step', default=2, type=int)
parser.add_argument('--encoding_alpha', default=1.0, type=float)
parser.add_argument('--pooling',
                    default='mean', choices=['mean', 'max', 'sum', 'median'])
parser.add_argument('--num_pool', default=4, type=int)
parser.add_argument('--time_window', default=250, type=int)
parser.add_argument('--twed_lambda', required=True, type=float)
parser.add_argument('--twed_nu', required=True, type=float)
parser.add_argument('--scale', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--split_encoding',
                    nargs='?', const=True, default=False, type=bool)
parser.add_argument('--part', required=True, type=int)
parser.add_argument('--num_parts', required=True, type=int)

args = parser.parse_args(sys.argv[1:])

training_lcs_file_path = args.training_lcs_file
clustering_result_file = args.clustering_file
dataset = args.dataset
s = args.time_step
encoding_alpha = args.encoding_alpha
pooling_method = args.pooling
t_w = args.time_window
twed_lambda = args.twed_lambda
twed_nu = args.twed_nu
scale_result = args.scale
split_encoding = args.split_encoding
num_pool = args.num_pool
part = args.part
num_parts = args.num_parts


def encode_lightcurve_twed(cluster_lcs, cluster_times, patches_lcs,
                           patches_times, alpha=1, split=True):
    num_patches = len(patches_lcs)
    num_centroids = len(cluster_lcs)
    D = np.zeros((num_patches, num_centroids))

    for i in xrange(num_patches):
        for j in xrange(num_centroids):
            A, A_times = patches_lcs[i], patches_times[i]
            B, B_times = cluster_lcs[j], cluster_times[j]
            D[i, j] = twed(A, A_times, B, B_times, lam=twed_lambda,
                           nu=twed_nu)

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


part_string = "encoding part {0} of {1}".format(part, num_parts)

if os.path.exists(os.getcwd()+"/"+clustering_result_file):
    with open(clustering_result_file, 'rb') as clustering_file:
        ap, centroid_paths, centroid_lcs, centroid_times = pickle.load(
            clustering_file)
else:
    print "{0}: Clustering file not found. This file is required.".format(
        part_string)
    sys.exit(2)

f = open(training_lcs_file_path, 'r')
training_lc_paths = [l[:-1] for l in f]

n = len(training_lc_paths)

"""
Cluster Pruning: to make encoding faster we only keep the exemplars which
have more than one member in the clustering.
"""
cluster_counts = ({i: ap.labels_.tolist().count(i) for i in
                  range(len(ap.cluster_centers_indices_))})
good_cluster_counts = {k: v for k, v in cluster_counts.iteritems() if v > 1}

centroid_times = [centroid_times[i] for i in good_cluster_counts.iterkeys()]
centroid_lcs = [centroid_lcs[i] for i in good_cluster_counts.iterkeys()]
centroid_paths = [centroid_paths[i] for i in good_cluster_counts.iterkeys()]


def standardize_lc(lc):
    mean, std = lc.mean(), lc.std()
    return (lc - mean) / std

print "Calculating new representation of the training dataset"
print datetime.datetime.now()
samples_per_lc = 1000
data_dict = {}
training_classes = []
training_ids = []
lcs_per_part = n/num_parts
begin_index = (part - 1)*lcs_per_part
if part == num_parts:
    end_index = n
else:
    end_index = lcs_per_part*part

for i in range(begin_index, end_index):
    print str(i)
    path = training_lc_paths[i]
    lc = lcu.open_lightcurve(path, dataset=dataset)
    lc = pp.eliminate_lc_noise(lc)
    # We are going to filter out lcs with too few observations
    # if len(lc) < 600:
    #    continue
    lc['mag'] -= lc['mag'].mean()
    lc['mag'] /= lc['mag'].std()
    lc.index = lc.index.values - lc.index.values[0]
    patch_mags = []
    patch_times = []
    training_classes.append(lcu.get_lc_class(path, dataset=dataset))
    training_ids.append(lcu.get_lightcurve_id(path, dataset=dataset))
    t_s = lc.index.values[0]
    for j in range(samples_per_lc):
        lc_s = lc[np.logical_and(lc.index > t_s, lc.index <= t_s + t_w)]
        if lc_s.shape[0] > 20:
            patch_mags.append(np.array(lc_s['mag'].tolist()))
            # make each patches time start from 0 for a meaningful comparison
            patch_times.append(lc_s.index.values - lc_s.index.values[0])
        t_s = t_s + s
    lc_df = encode_lightcurve_twed(centroid_lcs, centroid_times,
                                   patch_mags, patch_times, alpha=1, split=True)
    data_dict[i] = uflul.pool_lightcurve(lc_df, num_pool, method=pooling_method)


final_df = pd.DataFrame(data_dict).T

print "Encoding finished"
print datetime.datetime.now()

if scale_result:
    final_scaler = StandardScaler()
    final_df = pd.DataFrame(final_scaler.fit_transform(final_df))

final_df['id'] = training_ids
final_df['class'] = training_classes

print "shape final_df: " + repr(final_df.shape)
print "num clases final: " + str(final_df['class'].nunique())

filename = ("affinity_s{0}t_w{1}_alpha_{2}_pool_{3}_scale_{4}" +
            "_{5}_part{6}of{7}.csv").format(
    s, t_w, encoding_alpha, pooling_method,
    scale_result, dataset, part, num_parts)
# final_df.to_csv(("/n/home04/cmackenziek/msc-data/kmeans-ufl/results/" +
final_df.to_csv(("/Users/cristobal/Documents/msc-data/kmeans-ufl/results/" +
                 filename), index=False)
