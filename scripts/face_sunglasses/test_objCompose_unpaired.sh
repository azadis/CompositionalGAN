#!/bin/bash

# ========================================================
# Compositional GAN
# Script for testing the model with unpaired faces & sunglasses
# By Samaneh Azadi
# ========================================================

obj2=sunglasses
obj1=face

mode=train
name="${obj1}_${obj2}"

datalist="dataset/${name}/unpaired.txt"
datalist_t="dataset/${name}/test.txt"

exp_name="unpaired_compGAN"
name_train="${name}_train_${exp_name}"
name="${name}_test_${exp_name}"
dataset_mode='comp_decomp_unaligned'
batch_size=1
display_freq=1
print_freq=20
update_html_freq=5
n_latest=10000
loadSizeX=178
loadSizeY=178
lambda_L2=500
thresh1=0.99
thresh2=0.05
G1_comp=0
G2_comp=0
niter=2
niter_decay=2
LR=0.0002
EPOCH=60
how_many=200
display_port=8774
CUDA_ID=2


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

model_train="./checkpoints/${name_train}" 

cp ${model_train}/${EPOCH}_net_*.pth ./checkpoints/${name}/

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 test_composition.py --datalist ${datalist} --datalist_test ${datalist_t} \
								 --name ${name} --dataset_mode ${dataset_mode} --no_flip --conditional --decomp --img_completion \
								 --lr ${LR} --lambda_L2 ${lambda_L2} --G1_completion ${G1_comp} --G2_completion ${G2_comp} \
								 --loadSizeY ${loadSizeY} --loadSizeX ${loadSizeX} --batchSize ${batch_size} \
								 --niter ${niter} --niter_decay ${niter_decay} --how_many ${how_many} \
								 --which_epoch ${EPOCH} --Thresh1 ${thresh1} --Thresh2 ${thresh2} \
								 --print_freq ${print_freq} --update_html_freq ${update_html_freq} --n_latest ${n_latest}\
								 --display_port ${display_port} --display_freq ${display_freq} --eval