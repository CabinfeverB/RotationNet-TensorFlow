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

import network as MODEL
import dataset
from config import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch Size during training [default: 32]')

FLAGS = parser.parse_args()

BATCH_SIZE = config.BATCH_SIZE

HEIGHT = config.HEIGHT
WIDTH = config.WIDTH
VIEW_NUM = config.VIEW_NUM
CLASS_NUM = config.CLASS_NUM
LEARNING_RATE = config.LEARNING_RATE
VGG_CHECKPOINT_PATH = config.VGG_CHECKPOINT_PATH
MAX_EPOCH = config.MAX_EPOCH
DATA_DIR = config.DATA_DIR
MODEL_NAME = 'model2'

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train2.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

train_dataset = dataset.MultiViewDataset(DATA_DIR, 'train')
test_dataset = dataset.MultiViewDataset(DATA_DIR, 'test')
model = MODEL.Network()


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            [inptus_pl, label_pl] = model.get_inputs()
            is_training_pl = tf.placeholder(tf.bool, shape=())
            loss = model.inference([inptus_pl, label_pl], is_training_pl)
            train_end_points = model.get_tarin_collection()
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            train_op = optimizer.minimize(loss)

            saver = tf.train.Saver()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        tf_config.log_device_placement = False

        sess = tf.Session(config=tf_config)

        # saver.restore(sess, os.path.join(MODEL_DIR, 'model1.ckpt'))
        init = tf.global_variables_initializer()
        sess.run(init)
        #MODEL.load_vgg16_to_rotationnet(sess, VGG_CHECKPOINT_PATH)

        ops = {
            'inputs_pl': inptus_pl,
            'label_pl': label_pl,
            'end_points': train_end_points,
            'loss': loss,
            'is_training_pl': is_training_pl,
            'train_op': train_op
        }
        for epoch in range(MAX_EPOCH):
            log_string('******* EPOCH %03d *******' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops)


def train_one_epoch(sess, ops):
    is_training = True

    train_idxs = np.arange(0, len(train_dataset))
    num_batches = 32


    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data, batch_label, batch_gt = train_dataset.get_batches(
            train_idxs[start_idx:end_idx])
        feed_dict = {
            ops['inputs_pl']: batch_data,
            ops['label_pl']: batch_label,
            ops['is_training_pl']: is_training
        }

        _, end_points, _loss = sess.run(
            [ops['train_op'], ops['end_points'], ops['loss']],
            feed_dict=feed_dict)

        #scores = end_points['net_output']
        #scores = np.reshape(scores,[BATCH_SIZE,VIEW_NUM,VIEW_NUM,CLASS_NUM+1])
        #print(scores)
        #print('output', end_points['output'])
        #print('output_softmax', end_points['output_softmax'])
        #print('output_sub', end_points['output_sub'])
        #print('scores', end_points['scores'])
        #print('j_max', end_points['j_max'])
        #print('target_', end_points['target_'])
        pred_label = model.get_max_pred(end_points['output_softmax'])
        print(pred_label, batch_gt)
        log_string('loss: %f' % (_loss))


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

        pred_label = model.get_max_pred(end_points['net_output'])
        correct = np.sum(pred_label == batch_gt)
        total_correct += correct
        total_seen += BATCH_SIZE
        print(pred_label, batch_gt, batch_idx)
    log_string('test accuracy: %f' % (total_correct / float(total_seen)))
    log_string('test loss: %f' % (_loss))
    return total_correct / float(total_seen)


if __name__ == '__main__':
    train()
    LOG_FOUT.close()
