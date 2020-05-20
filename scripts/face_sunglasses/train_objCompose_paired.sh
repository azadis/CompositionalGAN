#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the model with paired faces & sunglasses
# By Samaneh Azadi
# ========================================================

obj2=sunglasses
obj1=face
mode=train
name="${obj1}_${obj2}"
datalist="dataset/${name}/paired.txt"
datalist_test="dataset/${name}/test.txt"


exp_name="paired_compGAN"
name="${name}_train_${exp_name}"
dataset_mode='comp_decomp_aligned'
batch_size=10
loadSizeX=178
loadSizeY=178
thresh1=0.99
thresh2=0.99
G1_comp=1
G2_comp=0
niter=300
niter_decay=300
which_epoch=0
which_epoch_completion=0

CUDA_ID=0
display_freq=550
print_freq=30
update_html_freq=550
save_epoch_freq=20
display_port=8774


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

LOG="./checkpoints/${name}/output.txt"

if [ -f $LOG ]; then
    rm $LOG
fi

exec &> >(tee -a "$LOG")

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 -u train_composition.py --datalist ${datalist} --datalist_test ${datalist_test} --decomp\
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan --conditional --no_flip --pool_size 0\
								 --loadSizeX ${loadSizeX} --loadSizeY ${loadSizeY} --batchSize ${batch_size}\
								 --niter ${niter} --niter_decay ${niter_decay} --which_epoch ${which_epoch} --epoch_count ${which_epoch} \
								 --Thresh1 ${thresh1} --Thresh2 ${thresh2}\
								 --display_port ${display_port} --print_freq ${print_freq}\
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq}  --save_epoch_freq ${save_epoch_freq} \

