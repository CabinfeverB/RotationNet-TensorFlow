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
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--case', type=str, default='1',
                    help='viewpoint case (1 or 2 or 3)')
parser.add_argument('-no', '--number', type=str, default='0',
                    help='experiment number')
parser.add_argument('-bitnot', '--bitnot', action='store_true',
                    help='cv2.bitwise_not')
parser.add_argument('-novggmean', '--novggmean', action='store_true',
                    help='no sub vgg_mean')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--center', action='store_true', help = 'center crop')
parser.add_argument('--max', type=float, default=0.970826, help='init max test '
                                                               'acc')

FLAGS = parser.parse_args()
PRETRAINED = FLAGS.pretrained
OPTIMIZER = FLAGS.optimizer
config.LEARNING_RATE = FLAGS.lr
config.CENTER = FLAGS.center
GPU_INDEX = FLAGS.gpu
if FLAGS.bitnot:
    config.BIT_NOT = True
if FLAGS.novggmean:
    config.VGG_MEAN = False
if FLAGS.case == '2':
    config.ALL_VIEW = False
    config.VIEW_NUM = 20
    config.DATA_DIR = '/unsullied/sharefs/jiangyongbo/data/MVCNN' \
                      '/modelnet40v2png/'
elif FLAGS.case == '1':
    config.VIEW_NUM = 12
    config.DATA_DIR = '/unsullied/sharefs/jiangyongbo/data/MVCNN' \
                      '/modelnet40v1png/'
elif FLAGS.case == '3':
    config.VIEW_NUM = 20
    config.DATA_DIR = '/unsullied/sharefs/jiangyongbo/data/MVCNN' \
                      '/modelnet40v3png/'
elif FLAGS.case == '4':
    config.VIEW_NUM = 20
    config.DATA_DIR = '/unsullied/sharefs/jiangyongbo/data/MVCNN' \
                      '/modelnet40v4png/'
elif FLAGS.case == '5':
    config.VIEW_NUM = 20
    config.DATA_DIR = '/unsullied/sharefs/jiangyongbo/data/MVCNN' \
                      '/modelnet40v2png_ori4'
else:
    config.VIEW_NUM = 20
    config.DATA_DIR = '/unsullied/sharefs/jiangyongbo/data/MVCNN' \
                      '/modelnet40v2png_ori4'

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
MODEL_NAME = 'model' + FLAGS.number

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train' + FLAGS.number + '.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

import network as MODEL
import dataset

train_dataset = dataset.MultiViewDataset(DATA_DIR, 'train')
test_dataset = dataset.MultiViewDataset(DATA_DIR, 'test')
model = MODEL.Network(FLAGS.case)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            [inptus_pl, label_pl] = model.get_inputs()
            is_training_pl = tf.placeholder(tf.bool, shape=())
            if FLAGS.case == '3' or FLAGS.case == '4' or FLAGS.case == '6':
                loss = model.inference_aligned([inptus_pl, label_pl],
                                               is_training_pl)
            else:
                loss = model.inference([inptus_pl, label_pl],
                                       is_training_pl)
            train_end_points = model.get_tarin_collection()
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
            else:
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            train_op = optimizer.minimize(loss)

            saver = tf.train.Saver()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
        maxx = FLAGS.max
        eval_one_epoch(sess, ops)
        for epoch in range(MAX_EPOCH):
            log_string('******* EPOCH %03d *******' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops)
            now = eval_one_epoch(sess, ops)

            if now > maxx:
                maxx = now
                save_path = saver.save(sess, os.path.join(
                    MODEL_DIR, MODEL_NAME + '.ckpt'))
                log_string(save_path)


def train_one_epoch(sess, ops):
    is_training = True

    train_idxs = np.arange(0, len(train_dataset))
    np.random.shuffle(train_idxs)
    num_batches = len(train_dataset) // BATCH_SIZE

    total_correct = 0
    total_seen = 0

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
        if FLAGS.case != '3' and FLAGS.case != '4' and FLAGS.case != '6':
            pred_label = model.get_max_pred(end_points['output_softmax'])
        else:
            pred_label = model.get_max_pred_case3(end_points['output_softmax'])
        correct = np.sum(pred_label == batch_gt)
        total_correct += correct
        total_seen += BATCH_SIZE
        print(pred_label, batch_gt, batch_idx)

    log_string('accuracy: %f' % (total_correct / float(total_seen)))
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

        if FLAGS.case != '3' and FLAGS.case != '4' and FLAGS.case != '6':
            pred_label = model.get_max_pred(end_points['output_softmax'])
        else:
            pred_label = model.get_max_pred_case3(end_points['output_softmax'])
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
