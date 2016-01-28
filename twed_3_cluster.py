import numpy as np
import pickle
import argparse
import os
import sys
from sklearn.cluster import AffinityPropagation
import datetime

print ' '.join(sys.argv)

parser = argparse.ArgumentParser(
    description='Cluster samples from lightcurves.')
parser.add_argument('--samples_file', required=True, type=str)
parser.add_argument('--pairwise_tweds_file', required=True, type=str)
parser.add_argument('--out_file', required=True, type=str)

args = parser.parse_args(sys.argv[1:])

samples_file = args.samples_file
pairwise_tweds_file = args.pairwise_tweds_file
cluster_result_file = args.out_file

if os.path.exists(os.getcwd()+"/"+cluster_result_file):
    print "Clustering result already saved"
    sys.exit(0)

if os.path.exists(os.getcwd()+"/"+samples_file):
    print "Opening lightcurve samples file..."
    with open(samples_file, 'rb') as s_file:
        paths, lcs, times = pickle.load(s_file)
else:
    print "Lightcurve samples file not found. This file is required."
    sys.exit(2)

if os.path.exists(pairwise_tweds_file+".npz"):
    print "Opening pairwise TWEDs matrix file..."
    with open(pairwise_tweds_file + ".npz", 'rb') as tweds_file:
        file = np.load(tweds_file)
        D = file['arr_0']
else:
    print "TWEDs file not found. This file is required."
    sys.exit(2)

affinities = -D

print "Starting Clustering..."
print datetime.datetime.now()
ap = AffinityPropagation(affinity='precomputed', verbose=True)
ap.fit(affinities)
print "Clustering done"
print datetime.datetime.now()

centroid_paths = [paths[x] for x in ap.cluster_centers_indices_]
centroid_lcs = [lcs[x] for x in ap.cluster_centers_indices_]
centroid_times = [times[x] for x in ap.cluster_centers_indices_]

print "Saving clustering result to {0}".format(cluster_result_file)
pickle.dump((ap, centroid_paths, centroid_lcs, centroid_times),
            open(cluster_result_file, 'w'),
            protocol=pickle.HIGHEST_PROTOCOL)
