#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the model with unpaired baskets & bottles
# By Samaneh Azadi
# ========================================================

obj2=bottle
obj1=basket
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
save_epoch_freq=50
lambda_mask=50
thresh1=0.8
thresh2=0.8
niter=300
niter_decay=300
niterSTN=50
niterCompletion=50
which_epoch=0
which_epoch_completion=0
display_port=8775
CUDA_ID=0


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

LOG="./checkpoints/${name}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi

exec &> >(tee -a "$LOG")


CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 -u train_composition.py --datalist ${datalist} --datalist_test ${datalist_test} \
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan --conditional --img_completion \
								 --pool_size 0 --batchSize ${batch_size}  --lambda_mask ${lambda_mask}  --no_flip  --decomp\
								 --niterCompletion ${niterCompletion} --which_epoch_completion ${which_epoch_completion} \
								 --niterSTN ${niterSTN} --niter ${niter} --niter_decay ${niter_decay} \
								 --which_epoch ${which_epoch} --epoch_count ${which_epoch} --Thresh1 ${thresh1} --Thresh2 ${thresh2}\
								 --display_port ${display_port}  --n_latest ${n_latest} --save_epoch_freq ${save_epoch_freq} \
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq} 

 