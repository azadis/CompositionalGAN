#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training our re-implementation of Appearance Flow network
# Given chairs, their masks, and azimuth angles for rotating chairs
# By Samaneh Azadi
# ========================================================

mode=train
obj=chair
datalist="dataset/single-objects/Shapenet/${obj}_${mode}_fullpath.txt"
masklist="dataset/single-objects/Shapenet/${obj}_mask_${mode}_fullpath.txt"
exp_name="AFN"
name="train_${exp_name}_${obj}"
batch_size=100
display_freq=5
print_freq=20
display_port=8557
update_html_freq=500
n_latest=5
niter=50
niter_decay=50
lr=0.0001
n_layers_D=4
lambda_masked=0.1
AFNmodel='fc'
CUDA_ID=0

if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 train_AFN.py --datalist ${datalist}  --display_port ${display_port}\
								 --masklist ${masklist} --name ${name} --no_flip\
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq} \
								 --lr ${lr} --batchSize ${batch_size} --n_layers_D ${n_layers_D} --n_latest ${n_latest}\
								 --niter ${niter} --niter_decay ${niter_decay} --AFNmodel ${AFNmodel} --lambda_masked ${lambda_masked}

 