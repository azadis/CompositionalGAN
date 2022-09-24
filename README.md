# Compositional GAN in PyTorch

This is the implementation of the Compositional GAN: Learning Image-Conditional Binary Composition. The code was written by [Samaneh Azadi](https://github.com/azadis). Please find the paper at [ArXiv](https://arxiv.org/pdf/1807.07560.pdf) or the [International Journal of Computer Vision 2020](https://link.springer.com/article/10.1007/s11263-020-01336-9?wt_mc=Internal.Event.1.SEM.ArticleAuthorOnlineFirst&utm_source=ArticleAuthorOnlineFirst&utm_medium=email&utm_content=AA_en_06082018&ArticleAuthorOnlineFirst_20200529).

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

- Download the datasets from [here](https://drive.google.com/drive/folders/1Ge9NrLnWnt2wIjLBoClEY_tIPd17LXGF?usp=sharing)
- For each pair of objects ```${obj1_obj2}``` in {chair_table, basket_bottle, city_car, face_sunglasses}, download the dataset by:

Individual chairs and tables are taken from [Shapenet dataset](https://www.shapenet.org/), faces from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and street scenes from [Cityscapes](https://www.cityscapes-dataset.com/). 


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
- Download our trained models on each binary composition task from [here](https://drive.google.com/drive/folders/1Ge9NrLnWnt2wIjLBoClEY_tIPd17LXGF?usp=sharing).

- To test your trained model or the above downloaded checkpoints, run
```bash
bash scripts/${obj1_obj2}/test_objCompose_paired.sh
``` 
or 
```
bash scripts/${obj1_obj2}/test_objCompose_unpaired.sh
```

- Before launching the above scripts, set ```display_port``` to an arbitrary port number ```${port}``` in the bash file and start the visdom server ```python -m visdom.server -p ${port}```.


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

## Citation
If you use this code or our compositional dataset, please cite our paper:
```
@article{azadi2018compositional,
  title={Compositional gan: Learning image-conditional binary composition},
  author={Azadi, Samaneh and Pathak, Deepak and Ebrahimi, Sayna and Darrell, Trevor},
  journal={arXiv preprint arXiv:1807.07560},
  year={2018}
}
```

