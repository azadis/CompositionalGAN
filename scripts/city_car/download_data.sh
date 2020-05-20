#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the city-car dataset
# By Samaneh Azadi
# ========================================================


if [ ! -d "dataset" ]; then
	mkdir "dataset"
fi

cd dataset
wget https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/dataset/city_car.tar.gz
tar -xvzf city_car.tar.gz
rm city_car.tar.gz