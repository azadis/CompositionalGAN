#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the chair-table dataset
# By Samaneh Azadi
# ========================================================


if [ ! -d "dataset" ]; then
	mkdir "dataset"
fi

cd dataset
wget --no-check-certificate https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/dataset/single-objects.tar.gz
tar -xvzf single-objects.tar.gz

wget --no-check-certificate https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/dataset/chair_table.tar.gz
tar -xvzf chair_table.tar.gz

rm single-objects.tar.gz
rm chair_table.tar.gz
