#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the bottle-basket dataset
# By Samaneh Azadi
# ========================================================


if [ ! -d "dataset" ]; then
	mkdir "dataset"
fi

cd dataset
wget --no-check-certificate https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/dataset/basket_bottle.tar.gz
tar -xvzf basket_bottle.tar.gz
rm basket_bottle.tar.gz
