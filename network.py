import tensorflow as tf
import numpy as np
import math
import sys
import os
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from config import config

HEIGHT = config.HEIGHT
WIDTH = config.WIDTH
CLASS_NUM = config.CLASS_NUM
STDDEV = config.STDDEV
BATCH_SIZE = config.BATCH_SIZE
VIEW_NUM = config.VIEW_NUM


class Network(object):
    def __init__(self, case):
        if case != '3' and case != '4' and case != '6':
            self.vcand = np.load('vcand_case' + '2' + '.npy')
        # self.vcand = np.array([[0,1],[1,0]])

    def get_inputs(self):
        inputs = []
        inputs.append(
            tf.placeholder(tf.float32, shape=(BATCH_SIZE, VIEW_NUM, HEIGHT,
                                              WIDTH, 3)))
        inputs.append(tf.placeholder(tf.int32, shape=(BATCH_SIZE, VIEW_NUM)))
        return inputs

    def inference(self, inputs, is_training):
        [views, target] = inputs
        views = tf.reshape(views, [-1, HEIGHT, WIDTH, 3])
        with tf.variable_scope('vgg_16', [views]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                    outputs_collections=end_points_collection):
                net = slim.repeat(views, 2, slim.conv2d, 64, [3, 3],
                                  scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                                  scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                                  scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                  scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                  scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID',
                                  scope='fc6')
                net = slim.dropout(net, 0.5, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                net = slim.dropout(net, 0.5, is_training=is_training,
                                   scope='dropout7')
                class_out = (CLASS_NUM + 1) * VIEW_NUM
                net = slim.conv2d(net, class_out, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                end_points[sc.name + '/fc8'] = net
                tf.add_to_collection(sc.name + '/fc8', net)
        vcand = tf.constant(self.vcand)
        output = tf.reshape(net, [-1, CLASS_NUM + 1])

        target = tf.reshape(target, [-1])
        target_ = tf.Variable(
            tf.ones(shape=[BATCH_SIZE * VIEW_NUM * VIEW_NUM], dtype=tf.int32))
        target_ = tf.multiply(target_, CLASS_NUM)

        tf.add_to_collection('output', output)
        output_ = tf.nn.softmax(output)
        tf.add_to_collection('output_softmax', output_)
        sub = tf.slice(output_, [0, CLASS_NUM],
                       [BATCH_SIZE * VIEW_NUM * VIEW_NUM, 1])
        tf.add_to_collection('sub', sub)
        output_ = tf.subtract(output_, sub)
        output_ = tf.slice(output_, [0, 0],
                           [BATCH_SIZE * VIEW_NUM * VIEW_NUM, CLASS_NUM])
        output_ = tf.reshape(output_, [-1, VIEW_NUM * VIEW_NUM, CLASS_NUM])
        output_ = tf.transpose(output_, [1, 2, 0])
        tf.add_to_collection('output_sub', output_)

        VCAND_NUM = vcand.get_shape().as_list()[0]
        scores = tf.zeros([CLASS_NUM, BATCH_SIZE], dtype=tf.float32)
        scores = self.get_one_score(scores, vcand, output_, 0)
        scores = tf.expand_dims(scores, 0)

        for i in range(1, VCAND_NUM):
            score = tf.zeros([CLASS_NUM, BATCH_SIZE], dtype=tf.float32)
            score = self.get_one_score(score, vcand, output_, i)
            score = tf.expand_dims(score, 0)
            scores = tf.concat([scores, score], 0)
        tf.add_to_collection('scores', scores)
        for n in range(BATCH_SIZE):
            j_max = tf.argmax(scores[:, target[n * VIEW_NUM], n], 0)
            tf.add_to_collection('j_max', j_max)
            for k in range(VIEW_NUM):
                target_ = self.set_value_1d(target_, n * VIEW_NUM * VIEW_NUM +
                                            vcand[j_max][k] * VIEW_NUM + k,
                                            target[n * VIEW_NUM])
        tf.add_to_collection('target_', target_)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                           labels=target_))
        return loss

    def inference_aligned(self, inputs, is_training):
        [views, target] = inputs
        views = tf.reshape(views, [-1, HEIGHT, WIDTH, 3])
        with tf.variable_scope('vgg_16', [views]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                    outputs_collections=end_points_collection):
                net = slim.repeat(views, 2, slim.conv2d, 64, [3, 3],
                                  scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                                  scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                                  scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                  scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                  scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID',
                                  scope='fc6')
                net = slim.dropout(net, 0.5, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                net = slim.dropout(net, 0.5, is_training=is_training,
                                   scope='dropout7')
                class_out = (CLASS_NUM + 1) * VIEW_NUM
                net = slim.conv2d(net, class_out, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                end_points[sc.name + '/fc8'] = net
                tf.add_to_collection(sc.name + '/fc8', net)

        output = tf.reshape(net, [-1, CLASS_NUM + 1])

        target = tf.reshape(target, [-1])
        target_ = tf.Variable(
            tf.ones(shape=[BATCH_SIZE * VIEW_NUM * VIEW_NUM], dtype=tf.int32))
        target_ = tf.multiply(target_, CLASS_NUM)

        tf.add_to_collection('output', output)
        output_ = tf.nn.softmax(output)
        tf.add_to_collection('output_softmax', output_)

        for i in range(BATCH_SIZE):
            for j in range(VIEW_NUM):
                target_ = self.set_value_1d(target_, i * VIEW_NUM * VIEW_NUM
                                            + j * VIEW_NUM + j, target[i *
                                                                    VIEW_NUM])

        tf.add_to_collection('target_', target_)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                           labels=target_))
        return loss

    def get_tarin_collection(self):
        ret = dict()
        ret['output_softmax'] = tf.add_n(tf.get_collection('output_softmax'))
        return ret

    def get_test_colllection(self):
        ret = dict()
        ret['output_softmax'] = tf.add_n(tf.get_collection('output_softmax'))
        return ret

    def set_value_1d(self, matrix, x, val):
        xxx = 1
        if type(xxx) != type(x):
            xx = x
        else:
            xx = tf.constant(x, dtype=tf.int64)
        w = int(matrix.get_shape()[0])
        val_diff = val - matrix[x]
        indices = tf.expand_dims(tf.expand_dims(xx, 0), 0)
        values = tf.expand_dims(val_diff, 0)
        diff_matrix = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=indices,
                            values=values,
                            dense_shape=[w]))
        return tf.add(matrix, diff_matrix)

    def get_one_score(self, score, vcand, output_, j):
        for k in range(VIEW_NUM):
            score = tf.add(score, output_[vcand[j, k] * VIEW_NUM + k])
        return score

    def get_max_pred(self, scoress):
        vcand = self.vcand
        VCAND_NUM = vcand.shape[0]
        scores = scoress.copy()
        scores = np.reshape(scores, [-1, (CLASS_NUM + 1)])
        for i in range(len(scores)):
            for k in range(CLASS_NUM):
                scores[i][k] = scores[i][k] - scores[i][CLASS_NUM]
                # scores[i][j * (CLASS_NUM + 1) + CLASS_NUM] = 0
        scores = np.reshape(scores, [-1, VIEW_NUM, VIEW_NUM * (CLASS_NUM + 1)])
        ret = []
        for _ in range(scores.shape[0]):
            s = np.zeros(CLASS_NUM * VCAND_NUM)
            for i in range(VCAND_NUM):
                for j in range(CLASS_NUM):
                    for k in range(VIEW_NUM):
                        idx = vcand[i][k]
                        s[i * CLASS_NUM + j] = s[i * CLASS_NUM + j] + scores[
                            _][k][
                            idx * (CLASS_NUM + 1) + j]
            ret.append(np.argmax(s) % CLASS_NUM)
        return np.array(ret)

    def get_max_pred_case3(self, scoress):
        scores = scoress.copy()
        scores = np.reshape(scores, [-1, VIEW_NUM, VIEW_NUM * (CLASS_NUM + 1)])
        ret = []
        for _ in range(scores.shape[0]):
            s = np.zeros(CLASS_NUM)
            for i in range(CLASS_NUM):
                for j in range(VIEW_NUM):
                    s[i] = s[i] + scores[_][j][j*(CLASS_NUM + 1) + i]
            ret.append(np.argmax(s))
        return np.array(ret)

def load_vgg16_to_rotationnet(sess, checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    param_map = {'conv1_1': 'vgg_16/conv1/conv1_1',
                 'conv1_2': 'vgg_16/conv1/conv1_2',
                 'conv2_1': 'vgg_16/conv2/conv2_1',
                 'conv2_2': 'vgg_16/conv2/conv2_2',
                 'conv3_1': 'vgg_16/conv3/conv3_1',
                 'conv3_2': 'vgg_16/conv3/conv3_2',
                 'conv3_3': 'vgg_16/conv3/conv3_3',
                 'conv4_1': 'vgg_16/conv4/conv4_1',
                 'conv4_2': 'vgg_16/conv4/conv4_2',
                 'conv4_3': 'vgg_16/conv4/conv4_3',
                 'conv5_1': 'vgg_16/conv5/conv5_1',
                 'conv5_2': 'vgg_16/conv5/conv5_2',
                 'conv5_3': 'vgg_16/conv5/conv5_3',
                 'fc6': 'vgg_16/fc6',
                 'fc7': 'vgg_16/fc7'}
    for name in param_map:
        load_param(sess, param_map[name],
                   reader.get_tensor(param_map[name] + '/weights'),
                   reader.get_tensor(param_map[name] + '/biases'))


def load_param(sess, name, data_w, data_b):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for subname, data in zip(('weights', 'biases'), (data_w, data_b)):
            var = tf.get_variable(subname)
            sess.run(tf.assign(var, data))


def test_load_param():
    with tf.Graph().as_default():
        inputs = tf.random_uniform((4, 12, 224, 224, 3))
        labels = tf.ones([4, 12], dtype=tf.int32)
        net = Network()
        pred = net.inference([inputs, labels], tf.constant(False))
        out = net.get_tarin_collection()
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        load_vgg16_to_rotationnet(sess,
                                  '/unsullied/sharefs/jiangyongbo/data/checkpoint/tensorflow/vgg/vgg_16.ckpt')
        res, end_point = sess.run([pred, out])
        print(
            res, end_point['net_output'],
            net.get_max_pred(end_point['net_output']))


if __name__ == '__main__':
    test_load_param()
