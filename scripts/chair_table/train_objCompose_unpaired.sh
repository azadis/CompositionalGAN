#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for training the model with unpaired chairs & tables
# By Samaneh Azadi
# ========================================================

obj2=table
obj1=chair
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
niter=300
niter_decay=300
niterSTN=60
niterCompletion=60
which_epoch=0
which_epoch_AFN=35000
which_epoch_completion=0
display_port=8774
CUDA_ID=1


if [ ! -d "./checkpoints/${name}" ]; then
	mkdir "./checkpoints/${name}"
fi

LOG="./checkpoints/${name}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi
# =======================================
##COPY pretrained AFN network from its corresponding directory
# =======================================
model_AFN_pretrained="./checkpoints/${obj1}_${obj2}_train_DOAFN_compose_b100" 
if [ ! -f "./checkpoints/${name}/35000_net_AFN.pth" ]; then
    cp "${model_AFN_pretrained}/35000_net_AFN.pth" "./checkpoints/${name}/"
fi

exec &> >(tee -a "$LOG")

CUDA_LAUNCH_BLOCKING=${CUDA_ID} CUDA_VISIBLE_DEVICES=${CUDA_ID} python3 -u train_composition.py --datalist ${datalist} --datalist_test ${datalist_test} --decomp --random_view\
								 --name ${name} --dataset_mode ${dataset_mode} --no_lsgan --conditional --img_completion --no_flip\
								 --pool_size 0 --lambda_mask ${lambda_mask}\
								 --batchSize ${batch_size}  --niterSTN ${niterSTN} --niterCompletion ${niterCompletion}\
								 --niter ${niter} --niter_decay ${niter_decay}\
								 --which_epoch ${which_epoch} --epoch_count ${which_epoch} --which_epoch_completion ${which_epoch_completion} \
								 --which_epoch_AFN ${which_epoch_AFN} --display_port ${display_port} --n_latest ${n_latest} --save_epoch_freq ${save_epoch_freq}\
								 --update_html_freq ${update_html_freq} --display_freq ${display_freq} --print_freq ${print_freq}

 