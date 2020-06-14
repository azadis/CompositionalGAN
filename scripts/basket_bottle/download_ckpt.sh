#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the basket-bottle checkpoints
# By Samaneh Azadi
# ========================================================


if [ ! -d "checkpoints" ]; then
	mkdir "checkpoints"
fi

cd checkpoints
wget --no-check-certificate https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/basket_bottle_train_paired_compGAN.tar.gz
tar -xvzf basket_bottle_train_paired_compGAN.tar.gz

wget --no-check-certificate  https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/basket_bottle_train_unpaired_compGAN.tar.gz
tar -xvzf basket_bottle_train_unpaired_compGAN.tar.gz


rm basket_bottle_train_paired_compGAN.tar.gz
rm basket_bottle_train_unpaired_compGAN.tar.gz
