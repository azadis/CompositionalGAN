#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the Relative Appearance Flow network
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
batch_size=100
display_freq=5
print_freq=20
update_html_freq=500
save_epoch_freq=5000
n_latest=5
niter=40000
niter_decay=10000
which_epoch=0
lambda_masked=0.1
display_port=8775
CUDA_ID=0

if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi


CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 train_AFN_compose.py --datalist ${datalist}\
								 --name ${name} --n_latest ${n_latest} --save_epoch_freq ${save_epoch_freq}\
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq} \
								 --batchSize ${batch_size} --no_flip\
								 --niter ${niter} --niter_decay ${niter_decay} --which_epoch ${which_epoch} --epoch_count ${which_epoch}\
								 --display_port ${display_port}



 