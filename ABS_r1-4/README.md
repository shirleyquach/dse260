# ABS


This is repository for paper ABS: Scanning Neural Networks for Back-doors by  Artificial Brain Stimulation. 

The repo contains two parts. The source code of ABS pytorch version used in TrojAI competition and the source code of ABS tensorflow+keras version.

## The source code of ABS pytorch version 

This repo include source code of ABS pytorch version. 

The source code of ABS pytorch version for TrojAI competition round 1-4 can be accessed at https://github.com/naiyeleo/ABS/blob/master/TrojAI_competition/

The ABS for TrojAI competition is enchanced for better performance. For example, during trigger reverse engineering besides the loss for stimulating compromised neuron we also include the loss that enlarge the target label's logits value.  


## The source code of ABS tensorflow+keras version

This repo also include the source code of ABS tensorflow+keras version

### Updated Dependences
Python 3.10, tensoflow=2.10, keras=2.10, imageio, numpy, pickle, h5py

Note: Tested with CUDA 12.1 and CUDNN 8.8

### Python 3.10 Environment Config
```Bash
conda create -y -n trojai_3 python=3.10
conda activate trojai_3
conda env update --file conda-requirements.yaml
```

### Original Dependences
Python 3.6, tensoflow=1.12.0, keras=2.2.4, imageio, numpy, pickle, h5py

### Original Environment Config
```Bash
conda create -y -n trojai python=3.6
conda activate trojai
conda env update --file conda-requirements.yaml
```
### File Description

You can edit `config.json` to change different models and settings for ABS. `models` contain 20 benign models and 21 compromised models. You can edit `config.json` to choose different models.

The seed images for CIFAR-10 dataset is in `cifar_seed_10.pkl` which contains 10 seed images, ABS reads in this file and perform analysis on these data. `cifar_seed_50.pkl` contains 50 seed images and running ABS on more images can increase stability.

The preprossing code of input images is written in `preprocess.py`. ABS calls `cifar.py` and to provide your own preprocess function, just change the code in `cifar.py`.


Currently, this version of ABS only work on CIFAR-10 dataset and may not support some structure. 
You can change the `abs.py` to work your structure.

Currently, ABS assumes the activation layer and conv/dense layer are seperated, i.e. the conv/dense layers do not have activation function and there is an activation layer after each dense/conv layer. 
Please refer to `reformat_model.py` to see how to seperate activation layers from conv/dense layer.

### Usage
**New**
`python abs_2.py`

**Old**
`python abs.py`

The program will output highest REASR for the model provided in `config.json`.
Triggers with over 80% REASR is shown `imgs` folder. `deltas` and `masks` store the numpy array for such triggers.

## Contacts

Yingqi Liu, liu1751@purdue.edu
