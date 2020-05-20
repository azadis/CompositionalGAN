#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the model with unpaired faces & sunglasses
# By Samaneh Azadi
# ========================================================

obj2=sunglasses
obj1=face
mode=train
name="${obj1}_${obj2}"

datalist="dataset/${name}/unpaired.txt"
datalist_test="dataset/${name}/test.txt"

exp_name="unpaired_compGAN"
name="${name}_train_${exp_name}"
dataset_mode='comp_decomp_unaligned'
batch_size=10
display_freq=550
print_freq=30
n_latest=5
update_html_freq=550
save_epoch_freq=20
loadSizeX=178
loadSizeY=178
thresh1=0.99
thresh2=0.05
G1_comp=1
G2_comp=0
lambda_mask=5
niter=30
niter_decay=30
niterSTN=5
niterCompletion=5
which_epoch=0
which_epoch_completion=0
display_port=8775
CUDA_ID=1


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

LOG="./checkpoints/${name}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi

exec &> >(tee -a "$LOG")

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 -u train_composition.py --datalist ${datalist} --datalist_test ${datalist_test}\
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan --conditional --img_completion --decomp\
								 --pool_size 0 --G1_completion ${G1_comp} --G2_completion ${G2_comp} \
								 --loadSizeY ${loadSizeY} --loadSizeX ${loadSizeX} --batchSize ${batch_size} --no_flip\
								 --lambda_mask ${lambda_mask}\
								 --Thresh1 ${thresh1} --Thresh2 ${thresh2}\
								 --niter ${niter} --niter_decay ${niter_decay} --niterCompletion ${niterCompletion} --niterSTN ${niterSTN}\
								 --which_epoch ${which_epoch} --epoch_count ${which_epoch}  --which_epoch_completion ${which_epoch_completion}\
								 --display_port ${display_port}  --n_latest ${n_latest} --save_epoch_freq ${save_epoch_freq}\
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq}

 