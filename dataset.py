# encoding: utf-8
'''
#author: Yongbo Jiang
#contact: cabinfeveroier@gmail.com
'''

import os
import numpy as np
import cv2

from config import config

CLASS_NUM = config.CLASS_NUM
VIEW_NUM = config.VIEW_NUM


class MultiViewDataset():

    def __init__(self, data_dir, data_type):
        self.x = []
        self.y = []
        self.data_dir = data_dir

        self.classes, self.class_to_idx = self.find_class(data_dir)

        for label in self.classes:
            cnt = 0
            all_views = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
            all_views.sort()
            views = []
            if config.ALL_VIEW == True:
                for item in all_views:
                    if item[-3:] != 'off':
                        views.append(os.path.join(label, data_type, item))
                        cnt += 1
                        if cnt % VIEW_NUM == 0:
                            self.x.append(views)
                            self.y.append(self.class_to_idx[label])
                            views = []
            else:
                for item in all_views:
                    if item[-3:] != 'off':
                        if cnt % 4 == 0:
                            views.append(os.path.join(label, data_type, item))
                        cnt += 1
                        if cnt % (VIEW_NUM*4) == 0:
                            self.x.append(views)
                            self.y.append(self.class_to_idx[label])
                            views = []

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):
        _views = self.x[indexx]
        views = []
        labels = []
        mean_bgr = (104, 116, 122)
        for view in _views:
            img = cv2.imread(os.path.join(self.data_dir, view))
            if config.BIT_NOT:
                img = cv2.bitwise_not(img)
            if config.VGG_MEAN:
                for c in range(3):
                    img[:, :, c] -= mean_bgr[c]
            if config.CENTER:
                top = int(config.HEIGHT / 2 - config.CROP_HEIGHT / 2)
                left = int(config.WIDTH / 2 - config.CROP_WIDTH / 2)
                right = left + config.CROP_WIDTH
                bottom = top + config.CROP_HEIGHT
                img = img[top:bottom, left:right, :]
                img = cv2.resize(img, (config.HEIGHT, config.WIDTH))
            views.append(img)
            labels.append(self.y[indexx])

        return np.array(views), np.array(labels), self.y[indexx]

    def get_batches(self, indexx):
        batch_num = indexx.shape[0]
        ret_x = []
        ret_y = []
        ret_l = []
        for _ in indexx:
            item_x, item_y, item_l = self.__getitem__(_)
            ret_x.append(item_x)
            ret_y.append(item_y)
            ret_l.append(item_l)

        return np.array(ret_x), np.array(ret_y), np.array(ret_l)


def label_matrix(label):
    batch_size = label.shape[0]
    ret = np.zeros([batch_size, CLASS_NUM])
    for i in range(batch_size):
        ret[i][label[i]] = 1
    return ret


def test_dataset():
    config.VIEW_NUM = 20
    config.DATA_DIR = '/unsullied/sharefs/jiangyongbo/data/MVCNN/modelnet40v3png/'
    dataset = MultiViewDataset(config.DATA_DIR, 'train')
    length = len(dataset)
    print(length)

    train_idxs = np.arange(0, len(dataset))
    #np.random.shuffle(train_idxs)
    num_batches = len(dataset) // 4

    total_correct = 0
    total_seen = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * 4
        end_idx = (batch_idx + 1) * 4

        batch_data, batch_label, batch_gt = dataset.get_batches(
            train_idxs[start_idx:end_idx])


if __name__ == '__main__':
    test_dataset()
