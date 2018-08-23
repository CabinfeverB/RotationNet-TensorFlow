# encoding: utf-8
'''
#author: Yongbo Jiang
#contact: cabinfeveroier@gmail.com
'''

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config():

    HEIGHT = 224
    WIDTH = 224
    VIEW_NUM = 12
    CLASS_NUM = 40

    STDDEV = 0.01
    LEARNING_RATE = 1e-5
    MOMENTUMU = 0.9
    BATCH_SIZE = 4
    MAX_EPOCH = 250
    ALL_VIEW = True

    BIT_NOT = False
    VGG_MEAN = True
    CENTER = False
    CROP_HEIGHT = 204
    CROP_WIDTH = 204

    VGG_CHECKPOINT_PATH = os.path.join(BATCH_SIZE, 'vgg16model', 'vgg_16.ckpt')
    DATA_DIR = os.path.join(BATCH_SIZE, 'data', 'modelnet40v2png_ori4')


config = Config()
