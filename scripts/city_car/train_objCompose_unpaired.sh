#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the model with unpaired street views & cars
# By Samaneh Azadi
# ========================================================

obj2=car
obj1=city
mode=train
name="${obj1}_${obj2}"

datalist="dataset/${name}/unpaired.txt"
datalist_test="dataset/${name}/test.txt"

exp_name="unpaired_compGAN"
name="${name}_train_${exp_name}"
dataset_mode='comp_decomp_unaligned'
batch_size=10
loadSizeY=256 #images are 128x256
fineSizeY=256
G1_comp=1
G2_comp=0
STN_model='deep'
lambda_mask=50
lr=0.00002
niter=100
niter_decay=100
niterSTN=100
niterCompletion=100
which_epoch=0
which_epoch_completion=0
which_epoch_STN=0

display_port=8775
display_freq=550
print_freq=30
update_html_freq=550
save_epoch_freq=50
CUDA_ID=2


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

LOG="./checkpoints/${name}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi

exec &> >(tee -a "$LOG")

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 -u train_composition.py --datalist ${datalist} --datalist_test ${datalist_test} --decomp \
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan  --conditional --img_completion --no_flip\
								 --pool_size 0  --niterCompletion ${niterCompletion} --which_epoch_completion ${which_epoch_completion} \
								 --fineSizeY ${fineSizeY} --loadSizeY ${loadSizeY}  --G1_completion ${G1_comp} --G2_completion ${G2_comp} \
								 --lambda_mask ${lambda_mask} --lr ${lr} --batchSize ${batch_size}\
								 --niterSTN ${niterSTN} --STN_model ${STN_model} --which_epoch_STN ${which_epoch_STN}\
								 --niter ${niter} --niter_decay ${niter_decay} --which_epoch ${which_epoch} --epoch_count ${which_epoch}\
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq}\
								 --display_port ${display_port} --save_epoch_freq ${save_epoch_freq} 

 