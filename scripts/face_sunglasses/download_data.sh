#!/bin/bash -f

# ========================================================
# Compositional GAN
# Script for downloading the face_sunglasses dataset
# By Samaneh Azadi
# ========================================================


if [ ! -d "dataset" ]; then
	mkdir "dataset"
fi

cd dataset
wget --no-check-certificate https://people.eecs.berkeley.edu/~sazadi/CompositionalGAN/dataset/face_sunglasses.tar.gz
tar -xvzf face_sunglasses.tar.gz
rm face_sunglasses.tar.gz
