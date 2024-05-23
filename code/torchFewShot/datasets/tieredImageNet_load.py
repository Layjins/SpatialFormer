from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import pickle
import cv2


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class tieredImageNet_load(object):
    """
    Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
    """
    dataset_dir = '/persons/jinxianglai/FewShotLearning/IMLN/tieredImagenet/data/tiered-imagenet/'

    def __init__(self, **kwargs):
        super(tieredImageNet_load, self).__init__()
        self.train_dir = os.path.join(self.dataset_dir, 'train_images_png.pkl')
        self.train_label_dir = os.path.join(self.dataset_dir, 'train_labels.pkl')
        self.val_dir = os.path.join(self.dataset_dir, 'val_images_png.pkl')
        self.val_label_dir = os.path.join(self.dataset_dir, 'val_labels.pkl')
        self.test_dir = os.path.join(self.dataset_dir, 'test_images_png.pkl')
        self.test_label_dir = os.path.join(self.dataset_dir, 'test_labels.pkl')

        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir, self.train_label_dir)
        #self.train, self.train_labels2inds, self.train_labelIds = self._process_dir2(self.train_dir, self.train_label_dir, self.val_dir, self.val_label_dir) # train+val
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir, self.val_label_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir, self.test_label_dir)

        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> TieredImageNet loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds),   len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds),  len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _get_pair(self, data, labels):
        assert (data.shape[0] == len(labels))
        data_pair = []
        for i in range(data.shape[0]):
            data_pair.append((data[i], labels[i]))
        return data_pair

    def _process_dir(self, file_path, label_path):
        dataset = load_data(file_path)
        images = np.zeros([len(dataset), 84, 84, 3], dtype=np.uint8)
        for ii, item in enumerate(dataset):
            im = cv2.imdecode(item, 1)
            images[ii] = im
        print(images.shape)
        #print(images[0].shape)
        dataset_label = load_data(label_path)
        labels = np.array(dataset_label['label_specific'])
        #print(labels.shape)
        data_pair = self._get_pair(images, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds

    def _process_dir2(self, file_path1, label_path1, file_path2, label_path2):
        dataset1 = load_data(file_path1)
        images1 = np.zeros([len(dataset1), 84, 84, 3], dtype=np.uint8)
        for ii, item in enumerate(dataset1):
            im = cv2.imdecode(item, 1)
            images1[ii] = im
        dataset_label1 = load_data(label_path1)
        labels1 = np.array(dataset_label1['label_specific'])

        dataset2 = load_data(file_path2)
        images2 = np.zeros([len(dataset2), 84, 84, 3], dtype=np.uint8)
        for ii, item in enumerate(dataset2):
            im = cv2.imdecode(item, 1)
            images2[ii] = im
        dataset_label2 = load_data(label_path2)
        labels2 = np.array(dataset_label2['label_specific']) + 351
        images = np.concatenate((images1, images2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        print(images.shape)
        data_pair = self._get_pair(images, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds

if __name__ == '__main__':
    tieredImageNet_load()
