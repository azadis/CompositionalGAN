#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the model with paired baskets & bottles
# By Samaneh Azadi
# ========================================================

obj2=bottle
obj1=basket

mode=train
name="${obj1}_${obj2}"

datalist="dataset/${name}/paired.txt"
datalist_test="dataset/${name}/test.txt"

exp_name="paired_compGAN"
name="${name}_train_${exp_name}"
dataset_mode='comp_decomp_aligned'
batch_size=10
niter=300
niter_decay=300
which_epoch=0
which_epoch_completion=0
thresh1=0.8
thresh2=0.8
pool_size=0
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

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 -u train_composition.py --datalist ${datalist} --datalist_test ${datalist_test} --decomp\
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan --conditional \
								 --batchSize ${batch_size} --no_flip --Thresh1 ${thresh1} --Thresh2 ${thresh2} --pool_size ${pool_size}\
								 --niter ${niter} --niter_decay ${niter_decay} --which_epoch ${which_epoch} --epoch_count ${which_epoch}\
								 --display_port ${display_port} --save_epoch_freq ${save_epoch_freq} \
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq}

 