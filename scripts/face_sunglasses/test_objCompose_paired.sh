#!/bin/bash

# ========================================================
# Compositional GAN
# Script for testing the model with paired faces & sunglasses
# By Samaneh Azadi
# ========================================================

obj2=sunglasses
obj1=face
mode=train
name="${obj1}_${obj2}"

datalist="dataset/${name}/paired.txt"
datalist_t="dataset/${name}/test.txt"

exp_name="paired_compGAN"

name_train="${name}_train_${exp_name}"
name="${name}_test_${exp_name}"
dataset_mode='comp_decomp_aligned'
batch_size=1
loadSizeX=178
loadSizeY=178
thresh1=0.99
thresh2=0.05
niter=2
niter_decay=2
LR=0.0002
EPOCH=600

n_latest=10000
display_freq=1
print_freq=20
update_html_freq=5
display_port=8774
CUDA_ID=2


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

model_train="./checkpoints/${name_train}" 

cp ${model_train}/${EPOCH}_net_*.pth ./checkpoints/${name}/

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 test_composition.py --datalist ${datalist} --datalist_test ${datalist_t} \
								 --name ${name} --dataset_mode ${dataset_mode} --Thresh1 ${thresh1} --Thresh2 ${thresh2} \
								 --lr ${LR} --no_flip --conditional --decomp --which_epoch ${EPOCH} --display_port ${display_port}\
								 --loadSizeY ${loadSizeY} --loadSizeX ${loadSizeX} --batchSize ${batch_size} --niter ${niter} --niter_decay ${niter_decay}\
								 --print_freq ${print_freq} --update_html_freq ${update_html_freq} --n_latest ${n_latest} --display_freq ${display_freq}\
								 --how_many 300 --eval\
								 
								 
