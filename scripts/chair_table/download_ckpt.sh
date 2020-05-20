#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the chair-table checkpoints
# By Samaneh Azadi
# ========================================================


if [ ! -d "checkpoints" ]; then
	mkdir "checkpoints"
fi

cd checkpoints
wget https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/chair_table_train_paired_compGAN.tar.gz
tar -xvzf chair_table_train_paired_compGAN.tar.gz

wget https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/chair_table_train_unpaired_compGAN.tar.gz
tar -xvzf chair_table_train_unpaired_compGAN.tar.gz


rm chair_table_train_paired_compGAN.tar.gz
rm chair_table_train_unpaired_compGAN.tar.gz