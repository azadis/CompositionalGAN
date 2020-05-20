#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for testing our implementation of Appearance Flow network
# Given chairs and azimuth angles to rotate chairs
# By Samaneh Azadi
# ========================================================

mode=test
obj=chair
datalist="dataset/single-objects/Shapenet/${obj}_${mode}_fullpath.txt"
masklist="dataset/single-objects/Shapenet/${obj}_mask_${mode}_fullpath.txt"
exp_name="AFN"
name="train_${exp_name}_${obj}"
batch_size=1
how_many=100
which_epoch=100
AFNmodel='fc'
lambda_masked=1
CUDA_ID=1


CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 test_AFN.py --datalist ${datalist} \
								 --masklist ${masklist} --name ${name} --no_flip\
								 --how_many ${how_many} --batchSize ${batch_size} \
								 --which_epoch ${which_epoch} \
								 --AFNmodel ${AFNmodel} --no_dropout --lambda_masked ${lambda_masked}

 