#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the model with paired chairs & tables
# By Samaneh Azadi
# ========================================================

obj2=table
obj1=chair

mode=train
name="${obj1}_${obj2}"

datalist="dataset/${name}/paired.txt"
datalist_test="dataset/${name}/test.txt"

exp_name="paired_compGAN"
name="${name}_train_${exp_name}"
dataset_mode='comp_decomp_aligned'
batch_size=10
niter=200
niter_decay=200
which_epoch=0
which_epoch_AFN=35000
which_epoch_completion=0

display_port=8770
display_freq=550
print_freq=30
update_html_freq=550
save_epoch_freq=50

CUDA_ID=1


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

LOG="./checkpoints/${name}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi
# =======================================
# COPY the pretrained AFN network from its corresponding directory
# =======================================
model_AFN_pretrained="./checkpoints/${obj1}_${obj2}_train_DOAFN_compose" 
if [ ! -f "./checkpoints/${name}/35000_net_AFN.pth" ]; then
    cp "${model_AFN_pretrained}/35000_net_AFN.pth" "./checkpoints/${name}/"
fi

exec &> >(tee -a "$LOG")

# --conditional --random_view --random_view_consistant --decomp --which_model_netG --continue_train --img_completion --rand_color

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 -u train_composition.py --datalist ${datalist} --datalist_test ${datalist_test} --decomp\
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan --conditional --random_view\
								 --batchSize ${batch_size} --no_flip\
								 --niter ${niter} --niter_decay ${niter_decay} --which_epoch ${which_epoch} --epoch_count ${which_epoch}\
								 --display_port ${display_port} --save_epoch_freq ${save_epoch_freq} --which_epoch_AFN  ${which_epoch_AFN}\
								 --display_id 1 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq}

 