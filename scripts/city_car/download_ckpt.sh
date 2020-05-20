#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the city-car checkpoints
# By Samaneh Azadi
# ========================================================


if [ ! -d "checkpoints" ]; then
	mkdir "checkpoints"
fi

cd checkpoints
wget https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/city_car_train_paired_compGAN.tar.gz
tar -xvzf city_car_train_paired_compGAN.tar.gz

wget https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/city_car_train_unpaired_compGAN.tar.gz
tar -xvzf city_car_train_unpaired_compGAN.tar.gz


rm city_car_train_paired_compGAN.tar.gz
rm city_car_train_unpaired_compGAN.tar.gz