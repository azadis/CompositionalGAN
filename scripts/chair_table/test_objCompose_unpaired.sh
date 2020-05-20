#!/bin/bash

# ========================================================
# Compositional GAN
# Script for testing the model with unpaired chairs & tables
# By Samaneh Azadi
# ========================================================
obj2=table
obj1=chair

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
lambda_L2=500
niter=2
niter_decay=2
LR=0.0002
EPOCH=600
G1_comp=0
G2_comp=0
which_epoch_AFN=35000
display_port=8774
CUDA_ID=2





if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

model_train="./checkpoints/${name_train}" 

cp ${model_train}/${EPOCH}_net_*.pth ./checkpoints/${name}/
if [ ! -f "./checkpoints/${name}/${which_epoch_AFN}_net_AFN.pth" ]; then
    cp ${model_train}/${which_epoch_AFN}_net_AFN.pth ./checkpoints/${name}/
fi



CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 test_composition.py --datalist ${datalist} --datalist_test ${datalist_t}\
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan --no_flip --decomp --conditional --random_view\
								 --lr ${LR} --lambda_L2 ${lambda_L2} --batchSize ${batch_size} \
								 --G1_completion ${G1_comp} --G2_completion ${G2_comp}\
								 --which_epoch_AFN ${which_epoch_AFN}  --which_epoch ${EPOCH}\
								 --niter ${niter} --niter_decay ${niter_decay}\
								 --how_many 300 --n_latest ${n_latest}\
								 --display_port ${display_port} --display_freq ${display_freq}\
								 --print_freq ${print_freq} --update_html_freq ${update_html_freq} --eval

