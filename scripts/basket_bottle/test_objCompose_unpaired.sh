#!/bin/bash

# ========================================================
# Compositional GAN
# Script for testing the model with unpaired baskets & bottles
# By Samaneh Azadi
# ========================================================
obj2=bottle
obj1=basket

mode=train
name="${obj1}_${obj2}"

##basket_bottle
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
lambda_L2=1000
lambda_mask=50
niter=1
niter_decay=1
LR=0.00001
EPOCH=600
thresh1=0.99
thresh2=0.99
G1_comp=0
G2_comp=0
display_port=8775
CUDA_ID=0





if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

model_train="./checkpoints/${name_train}" 

cp ${model_train}/${EPOCH}_net_*.pth ./checkpoints/${name}/


CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 test_composition.py --datalist ${datalist} --datalist_test ${datalist_t} \
								 --name ${name} --dataset_mode ${dataset_mode} --no_flip --conditional --decomp\
								 --lr ${LR} --lambda_L2 ${lambda_L2} --lambda_mask ${lambda_mask} --batchSize ${batch_size} \
								 --G1_completion ${G1_comp} --G2_completion ${G2_comp}\
								 --niter ${niter} --niter_decay ${niter_decay} --Thresh1 ${thresh1} --Thresh2 ${thresh2}\
								 --which_epoch ${EPOCH} --display_port ${display_port}\
								 --print_freq ${print_freq} --update_html_freq ${update_html_freq} --n_latest ${n_latest}\
								 --display_port ${display_port} --how_many 300 --eval

