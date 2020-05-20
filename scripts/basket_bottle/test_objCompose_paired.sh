#!/bin/bash

# ========================================================
# Compositional GAN
# Script for testing the model with paired baskets & bottles
# By Samaneh Azadi
# ========================================================

obj2=bottle
obj1=basket
mode=train
name="${obj1}_${obj2}"

datalist="dataset/${name}/paired.txt"
datalist_t="dataset/${name}/test.txt"
exp_name="paired_compGAN"

name_train="${name}_train_${exp_name}"
name="${name}_test_${exp_name}"
dataset_mode='comp_decomp_aligned'
batch_size=1
display_freq=1
print_freq=20
update_html_freq=5
display_port=8774
niter=2
niter_decay=2
LR=0.00002
EPOCH=600
CUDA_ID=1


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

model_train="./checkpoints/${name_train}" 

cp ${model_train}/${EPOCH}_net_*.pth ./checkpoints/${name}/


CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 test_composition.py --datalist ${datalist} --datalist_test ${datalist_t}\
								 --name ${name} --dataset_mode ${dataset_mode} --no_flip --conditional --decomp\
								 --lr ${LR} --batchSize ${batch_size}  --how_many 300 \
								 --niter ${niter} --niter_decay ${niter_decay} --which_epoch ${EPOCH} --display_port ${display_port}\
								 --print_freq ${print_freq} --update_html_freq ${update_html_freq} --display_freq ${display_freq} \
								 --eval