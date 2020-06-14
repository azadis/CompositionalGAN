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

wget --no-check-certificate  https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/ckpts/city_car_train_unpaired_compGAN.tar.gz
tar -xvzf city_car_train_unpaired_compGAN.tar.gz


rm city_car_train_unpaired_compGAN.tar.gz
