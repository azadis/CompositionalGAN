#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for testing the Relative Appearance Flow network
# Given chairs and table masks
# By Samaneh Azadi
# ========================================================

obj2=table
obj1=chair

mode=train
name="${obj1}_${obj2}"
datalist="dataset/single-objects/Shapenet/${obj1}_${obj2}_${mode}_halfpath.txt"

exp_name="DOAFN_compose"
name="${name}_train_${exp_name}"
batch_size=1
how_many=100
which_epoch=35000
init_type='normal'
lambda_masked=0.1
CUDA_ID=1


CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 test_AFN_compose.py --datalist ${datalist}\
								 --name ${name} --batchSize ${batch_size} --no_flip\
								 --which_epoch ${which_epoch}\
