#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the face-sunglasses checkpoints
# By Samaneh Azadi
# ========================================================


if [ ! -d "checkpoints" ]; then
	mkdir "checkpoints"
fi

cd checkpoints
wget https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/face_sunglasses_train_paired_compGAN.tar.gz
tar -xvzf face_sunglasses_train_paired_compGAN.tar.gz

wget https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/face_sunglasses_train_unpaired_compGAN.tar.gz
tar -xvzf face_sunglasses_train_unpaired_compGAN.tar.gz


rm face_sunglasses_train_paired_compGAN.tar.gz
rm face_sunglasses_train_unpaired_compGAN.tar.gz