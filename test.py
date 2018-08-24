# encoding: utf-8
'''
#author: Yongbo Jiang
#contact: cabinfeveroier@gmail.com
'''

import argparse
import math
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

from config import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch Size during training [default: 4]')
parser.add_argument('--pretrained', type=str, default='-1',
                    help='use pre-trained model')
parser.add_argument('--case', type=str, default='1',
                    help='viewpoint case 1 or 2 ')
parser.add_argument('-bitnot', '--bitnot', action='store_true',
                    help='cv2.bitwise_not')
parser.add_argument('-novggmean', '--novggmean', action='store_true',
                    help='no sub vgg_mean')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--center', action='store_true', help = 'center crop')

FLAGS = parser.parse_args()
PRETRAINED = FLAGS.pretrained
config.CENTER = FLAGS.center
GPU_INDEX = FLAGS.gpu
if FLAGS.bitnot:
    config.BIT_NOT = True
if FLAGS.novggmean:
    config.VGG_MEAN = False

BATCH_SIZE = config.BATCH_SIZE

HEIGHT = config.HEIGHT
WIDTH = config.WIDTH
VIEW_NUM = config.VIEW_NUM
CLASS_NUM = config.CLASS_NUM
LEARNING_RATE = config.LEARNING_RATE
VGG_CHECKPOINT_PATH = config.VGG_CHECKPOINT_PATH
MAX_EPOCH = config.MAX_EPOCH
DATA_DIR = config.DATA_DIR
MOMENTUM = config.MOMENTUMU
import network as MODEL
import dataset

test_dataset = dataset.MultiViewDataset(DATA_DIR, 'test')
model = MODEL.Network(FLAGS.case)


def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            [inptus_pl, label_pl] = model.get_inputs()
            is_training_pl = tf.placeholder(tf.bool, shape=())
            if FLAGS.case == '2':
                loss = model.inference_aligned([inptus_pl, label_pl],
                                               is_training_pl)
            else:
                loss = model.inference([inptus_pl, label_pl],
                                       is_training_pl)
            train_end_points = model.get_tarin_collection()
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            train_op = optimizer.minimize(loss)

            saver = tf.train.Saver()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        tf_config.log_device_placement = False

        sess = tf.Session(config=tf_config)

        if PRETRAINED != '-1':
            saver.restore(sess, os.path.join(MODEL_DIR, 'model' + PRETRAINED +
                                             '.ckpt'))
            print('load pretrained model')
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            MODEL.load_vgg16_to_rotationnet(sess, VGG_CHECKPOINT_PATH)

        ops = {
            'inputs_pl': inptus_pl,
            'label_pl': label_pl,
            'end_points': train_end_points,
            'loss': loss,
            'is_training_pl': is_training_pl,
            'train_op': train_op
        }
        eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False

    test_idxs = np.arange(0, len(test_dataset))
    np.random.shuffle(test_idxs)
    num_batches = len(test_dataset) // BATCH_SIZE

    total_correct = 0
    total_seen = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data, batch_label, batch_gt = test_dataset.get_batches(
            test_idxs[start_idx:end_idx])
        feed_dict = {
            ops['inputs_pl']: batch_data,
            ops['label_pl']: batch_label,
            ops['is_training_pl']: is_training
        }

        end_points, _loss = sess.run(
            [ops['end_points'], ops['loss']], feed_dict=feed_dict)

        if FLAGS.case == '1':
            pred_label = model.get_max_pred(end_points['output_softmax'])
        else:
            pred_label = model.get_max_pred_aligned(end_points['output_softmax'])
        correct = np.sum(pred_label == batch_gt)
        total_correct += correct
        total_seen += BATCH_SIZE
    print('test accuracy: %f' % (total_correct / float(total_seen)))
    return total_correct / float(total_seen)


if __name__ == '__main__':
    test()
