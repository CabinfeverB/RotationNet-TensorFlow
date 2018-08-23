# Introduction

This is an unofficial inplementation of [
RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints](https://arxiv.org/abs/1603.06208) in TensorFlow.

# Dependencies

- `python3.5+`
- `TensorFlow` (tested on 1.8)
- `opencv`

# Installation

Just clone this repository.

# Data Preparation

Download multi-view images 

    $ bash get_full_modelnet_png.sh

# Pretrained model preparation

Down VGG16 pre-trained models which has been trained on the ILSVRC-2012-CLS 
image classification dataset from TensorFlow

    $ bash get_vgg16_model.sh

# Train

If you want to train the model w/o viewpoint prediction

    # N is your checkpoint iteration
    $ python3 train.py --case 2 --center --lr 0.00001 -no N
 
If you want to train the model with viewpoint prediction

    # N is your checkpoint iteration
    $ python3 train.py --case 1 --center --lr 0.00001 -no N
    
If you want to fine-tune model No. 0

    # N is your checkpoint iteration
    $ python3 train.py --case 2 --center --lr 0.00001 --pretrained N

# Evaluate

    # N is your checkpoint iteration
    $ python3 test.py --case 1 --center --pretrained N

# Performances


| ModelNet40 | with viewpoint prediction | w/o viewpoint prediction |
|:-:|:-:|:-:|
| Reported | 97.37 | null |
| Reproduced | 97.2024  | 96.72  |
