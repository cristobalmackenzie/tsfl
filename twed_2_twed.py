# path hack to be able to import /homebrew/python/numba
import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
from numba import jit
from twed import twed
import datetime

print ' '.join(sys.argv)

parser = argparse.ArgumentParser(
    description='Get samples from lightcurves.')
parser.add_argument('--twed_lambda', required=True, type=float)
parser.add_argument('--twed_nu', required=True, type=float)
parser.add_argument('--samples_file', required=True, type=str)
parser.add_argument('--out_file', required=True, type=str)
parser.add_argument('--part', required=True, type=int)
parser.add_argument('--num_parts', required=True, type=int)

args = parser.parse_args(sys.argv[1:])

twed_lambda = args.twed_lambda
twed_nu = args.twed_nu
samples_file = args.samples_file
pairwise_tweds_file = args.out_file
part = args.part
num_parts = args.num_parts

if os.path.exists(os.getcwd()+"/"+pairwise_tweds_file):
    print "Pairwise TWEDs matrix already saved"
    sys.exit(0)

if os.path.exists(os.getcwd()+"/"+samples_file):
    print "Opening lightcurve samples file..."
    with open(samples_file, 'rb') as s_file:
        paths, lcs, times = pickle.load(s_file)
else:
    print "Lightcurve samples file not found. This file is required."
    sys.exit(2)


def complexity_coeff(lc_a, times_a, lc_b, times_b):
    complexity_1 = np.sum(np.sqrt(np.power(np.diff(lc_a), 2)) +
                          np.sqrt(np.power(np.diff(times_a), 2)))
    complexity_2 = np.sum(np.sqrt(np.power(np.diff(lc_b), 2)) +
                          np.sqrt(np.power(np.diff(times_b), 2)))
    return max(complexity_1, complexity_2)/min(complexity_1, complexity_2)


def get_coords(N, num_twed):
    """
    This function returns the coordinates corresponding to the num_twed.
    TODO: explain well
    """
    i, j = 0, 0
    sub = N
    while num_twed > sub:
        num_twed -= sub
        sub -= 1
        i += 1
    return (i, i + num_twed - 1)


def pairwise_tweds(pairwise_tweds_file, lcs, times, part, num_parts,
                   lam=0.5, nu=1e-5):
    """
    For parallelization in many jobs, we calculate how many pairwise distances
    we need to calculate, and then figure out the border indexes of the ones we
    need to do in the current job
    """
    N = len(lcs)
    num_tweds = N*(N + 1)/2
    tweds_per_part = num_tweds/num_parts
    begin_index = (part - 1)*tweds_per_part
    if part == num_parts:
        end_index = num_tweds
    else:
        end_index = tweds_per_part*part
    print "N: {0}".format(N)
    print "num_tweds: {0}".format(num_tweds)
    print "tweds_per_part: {0}".format(tweds_per_part)
    print "begin_index: {0}".format(begin_index)
    print "end_index: {0}".format(end_index)
    print "part: {0}".format(part)
    print "num_parts: {0}".format(num_parts)
    D = np.zeros((N, N))
    """
    twed_file = h5py.File(pairwise_tweds_file + ".hdf5", "w")
    D = twed_file.create_dataset("twed_matrix_part", (tweds_per_part, ),
                                 compression="gzip",
                                 scaleoffset=10,
                                 shuffle=True)
    D.attrs["N"] = N
    D.attrs["part"] = part
    D.attrs["num_parts"] = num_parts
    """
    k = 0
    for current_twed in xrange(begin_index, end_index):
        i, j = get_coords(N, current_twed + 1)
        # starting from i saves time since matrix is symmetric
        # if k % 1000 == 0:
            # print "{0} of {1}".format(k, tweds_per_part)
        twed_val = twed(lcs[i], times[i], lcs[j], times[j], lam=lam, nu=nu)
        complex_coeff = complexity_coeff(lcs[i], times[i], lcs[j], times[j])
        final_val = twed_val*complex_coeff
        # D[current_twed - begin_index] = twed_val
        D[i, j] = final_val
        D[j, i] = final_val
        k += 1
    # return twed_file
    return D

# file = pairwise_tweds(pairwise_tweds_file, lcs, times, part, num_parts,
#                      lam=twed_lambda, nu=twed_nu)
print "Calculating pairwise TWEDs"
print datetime.datetime.now()
D = pairwise_tweds(pairwise_tweds_file, lcs, times, part, num_parts,
                   lam=twed_lambda, nu=twed_nu)

print "Saving TWED matrix to {0}".format(pairwise_tweds_file)
print datetime.datetime.now()
# file.close()
np.savez_compressed(pairwise_tweds_file, D)
