# Compositional GAN in PyTorch

This is the implementation of the [Compositional GAN: Learning Image-Conditional Binary Composition](https://arxiv.org/pdf/1807.07560.pdf). The code was written by [Samaneh Azadi](https://github.com/azadis).
If you use this code or our dataset for your research, please cite:

Compositional GAN: Learning Image-Conditional Binary Composition; [Samaneh Azadi](https://people.eecs.berkeley.edu/~sazadi/), [Deepak Pathak](https://people.eecs.berkeley.edu/~pathak/), [Sayna Ebrahimi](https://saynaebrahimi.github.io), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), in arXiv, 2018.

## Prerequisites:
- Linux or macOS
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN

## Preparation
### Installation
- Install PyTorch 1.0 and dependencies from http://pytorch.org
- Install Torch vision from the source:
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

- Install [visdom](https://github.com/facebookresearch/visdom) python library: 
```
pip install visdom
```

- Clone this repo:
```bash
git clone https://github.com/azadis/CompositionalGAN
cd CompositionalGAN
```

### Datasets
- For each pair of objects ```${obj1_obj2}``` in {chair_table, basket_bottle, city_car, face_sunglasses}, download the dataset by:
```bash
bash scripts/${obj1_obj2}/download_data.sh
```


## Training
### Viewpoint Transformation module:
If your model includes viewpoint transformation as in the chair_table experiment, train the Appearance Flow Network (AFN) by:
```
bash scripts/chair_table/train_AFN_Compose.sh
```
or download our trained AFN model:
```
bash scripts/chair_table/download_ckpt.sh
```

### Paired Data
- To train a compositional GAN model in order to compose each pair of objects ```${obj1_obj2}``` given a paired training data, do: 
```
bash scripts/${obj1_obj2}/train_objCompose_paired.sh
```

- Before launching the above training script, set ```display_port``` to an arbitrary port number ```${port}``` in the bash file and start the visdom server ```python -m visdom.server -p ${port}```.

### Unpaired Data
- To train a model with unpaired training data, follow the same steps as above:
 ```bash
  scripts/${obj1_obj2}/train_objCompose_unpaired.sh
  ```

## Testing
- To download our trained models on each binary composition task of ```${obj1_obj2}```:
```
bash scripts/${obj1_obj2}/download_ckpt.sh
```

- To test your trained model or the above downloaded checkpoints, run
```bash
bash scripts/${obj1_obj2}/test_objCompose_paired.sh
``` 
or 
```
bash scripts/${obj1_obj2}/test_objCompose_unpaired.sh
```


## Visualization
- To visualize the test results for the ${obj1_obj2} paired training after ```${epoch}``` epochs:
- Before ESMR 
```bash
cd results/${obj1_obj2}_test_paired_compGAN/test_${epoch}/
python -m http.server 8884
```

- After ESMR 
```bash
cd results/finetune/${obj1_obj2}_test_paired_compGAN/test_${epoch}/
python -m http.server 8884
```

- Replace ```paired``` with ```unpaired``` if you are training under the latter scenario.
- Then in your local machine: ```ssh -N -f -L localhost:8884:localhost:8884 remote_user@remote_host```

