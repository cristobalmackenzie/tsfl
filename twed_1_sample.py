import lightcurves.uflutil as uflul
import lightcurves.preprocess as pp
import lightcurves.lightcurveutil as lcu
import numpy as np
import pandas as pd
from random import random
from random import shuffle
import pickle
import os
import sys
import argparse
import datetime

print ' '.join(sys.argv)


def standardize_lc(lc):
    mean, std = lc.mean(), lc.std()
    return (lc - mean) / std

parser = argparse.ArgumentParser(
    description='Get samples from lightcurves.')
parser.add_argument('--lc_paths_file', required=True, type=str)
parser.add_argument('--time_window', required=True, type=int)
parser.add_argument('--num_samples', required=True, type=int)
parser.add_argument('--out_file', required=True, type=str)
parser.add_argument('--dataset', default='macho',
                    choices=['macho', 'ogle', 'eros'])


args = parser.parse_args(sys.argv[1:])

num_samples = args.num_samples
t_w = args.time_window
paths_file = args.lc_paths_file
samples_file = args.out_file
dataset = args.dataset

if os.path.exists(os.getcwd()+"/"+samples_file):
    print "Lightcurve samples already saved :-)"
    sys.exit(0)

lc_paths = open(paths_file, 'r').readlines()
lc_paths = map(lambda x: x[:-1], lc_paths)  # remove newline char

classes = lcu.get_dataset_classes(dataset=dataset)

# We want to get the same number of samples from each class
num_per_class = num_samples/len(classes)

print datetime.datetime.now()
samples = []
paths = []

for lc_class in classes:
    print "Sampling "+lc_class
    # Get all the lightcurve file paths for a given class
    class_paths = [path for path in lc_paths if lc_class in path]
    class_lcs = lcu.open_lightcurves(class_paths, dataset=dataset)
    # class_lcs = map(pp.eliminate_lc_noise, class_lcs)
    samples_taken_of_current_class = 0
    # if we got no lightcurves from current class, skip !!
    if len(class_lcs) == 0:
        continue
    samples_per_lc = num_per_class/len(class_lcs) + 1
    for i in range(len(class_lcs)):
        lc = class_lcs[i]
        # get all lightcurves to have zero mean and unit variance
        lc['mag'] -= lc['mag'].mean()
        lc['mag'] /= lc['mag'].std()
        path = class_paths[i]
        samples_taken = 0
        num_errors = 0
        while samples_taken < samples_per_lc and num_errors < 10:
            # sample a piece of the lightcurve dataframe, of time length t_w
            t_min, t_max = min(lc.index), max(lc.index)
            t_sample = t_min + (t_max - t_min - t_w)*random()
            lc_s = lc[np.logical_and(lc.index > t_sample,
                                     lc.index <= t_sample + t_w)]
            if len(lc_s) < 20:
                num_errors += 1
                # print "too little observations"
                continue
            left_time_gap = lc_s.index[0] - t_sample
            right_time_gap = t_sample + t_w - lc_s.index[-1]
            middle_time_gap = max(np.diff(lc_s.index.tolist()))
            max_time_gap = max(left_time_gap, right_time_gap, middle_time_gap)
            if len(lc_s) > 20 and max_time_gap < 50:
                samples.append(lc_s)
                paths.append(path)
                samples_taken += 1
                samples_taken_of_current_class += 1
            else:
                # print "too little observations"
                num_errors += 1
        if samples_taken_of_current_class > num_per_class:
            break


# get the timestamps as numpy arrays
times = map(lambda x: np.array(x.index.tolist()), samples)
# make times start from 0
times = map(lambda x: x - x[0], times)
# get the magnitudes as lists
lcs = map(lambda x: x['mag'], samples)
lcs = [x.as_matrix() for x in lcs]

print "Took {0} lightcurve samples".format(len(samples))
print datetime.datetime.now()
print "Saving samples file to {0}".format(samples_file)
pickle.dump((paths, lcs, times), open(samples_file, 'w'),
            protocol=pickle.HIGHEST_PROTOCOL)
