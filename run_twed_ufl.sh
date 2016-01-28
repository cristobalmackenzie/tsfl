#!/bin/bash

# .txt file with lightcurve file paths
LCS_FILE="/path/to/txt/file/with/lightcurve/file/paths"
# time window length in days
TIME_WINDOW=250
# time step in days
STEP=10
# number of sample to take for clustering
NUM_SAMPLES=20000
# pooling method: mean, max and median are implemented
POOLING_METHOD="max"
# number of regions over which to perform pooling
NUM_POOL=4
# dataset being used:
# this is for the open_lightcurve function which
# considers each datasets different file format. macho, ogle and eros are
# implemented, and new datasets should take care of implementing the
# open_lightcurve routine and adding a dataset name to the allowed arguments
# in
DATASET="macho"
# parameters for the Time Warp Edit Distance
TWED_LAMBDA=0.5
TWED_NU=0.00001


BASE_DATA_PATH="/path/where/you/want/to/store/data"
SAMPLES_FILE="${BASE_DATA_PATH}lcs_samples_t_w=${TIME_WINDOW}_num${NUM_SAMPLES}_${DATASET}.pickle"
TWED_MATRIX_FILE="${BASE_DATA_PATH}twed_matrix_t_w=${TIME_WINDOW}_num${NUM_SAMPLES}_${DATASET}"
CLUSTER_RESULT_FILE="${BASE_DATA_PATH}cluster_result_t_w=${TIME_WINDOW}_num${NUM_SAMPLES}_${DATASET}.pickle"

# these parameters are useful when trying to encode a big training set
# the encoding step can be divided into as many tasks as possible, useful to
# divide the task in a SLURM job array
PART=1
NUM_PARTS=1

TWED_MATRIX_PART_FILE="${BASE_DATA_PATH}${PART}of${NUM_PARTS}${TWED_MATRIX_FILE}"

# args: --lc_paths_file --time_window --num_samples --num_samples_training --out_file --dataset
python twed_1_sample.py --lc_paths_file=$LCS_FILE --time_window=$TIME_WINDOW --num_samples=$NUM_SAMPLES --out_file=$SAMPLES_FILE --dataset=$DATASET

# args: --twed_lambda --twed_nu --samples_file --out_file
python twed_2_twed.py --twed_lambda=$TWED_LAMBDA --twed_nu=$TWED_NU --samples_file=$SAMPLES_FILE --out_file=$TWED_MATRIX_PART_FILE --part=$PART --num_parts=$NUM_PARTS

# args: --samples_file --pairwise_tweds_file --out_file
python twed_3_cluster.py --samples_file=$SAMPLES_FILE --pairwise_tweds_file=$TWED_MATRIX_PART_FILE --out_file=$CLUSTER_RESULT_FILE

# args: --training_lcs_file --cluster_result_file --dataset --time_step --encoding_alpha --pooling
# ... --num_pool --time_window --scale --split_encoding --part --num_parts
python twed_4_encode.py --training_lcs_file=$TRAINING_LCS_FILE --clustering_file=$CLUSTER_RESULT_FILE  --dataset=$DATASET --time_step=$STEP --encoding_alpha=$ENCODING_ALPHA --pooling=$POOLING_METHOD --num_pool=$NUM_POOL --time_window=$TIME_WINDOW --twed_lambda=$TWED_LAMBDA --twed_nu=$TWED_NU --part=$PART --num_parts=$NUM_PARTS
