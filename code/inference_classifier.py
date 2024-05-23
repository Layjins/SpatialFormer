from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import numpy as np
import random
import cv2
from shutil import copy

import torch
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
sys.path.append('./torchFewShot')

from torchFewShot.transforms import transforms as T

from torchFewShot.models.net import Model
from torchFewShot.models.gcn import SCGCN_self
from torchFewShot.models.classifier_finetune import ClasifierFinetune
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import adjust_learning_rate
from torchFewShot.utils.mkdir import check_mkdir, check_makedirs
from torchFewShot.utils.tensor_to_img import imageSavePIL
from tensorboardX import SummaryWriter


# import config for test
from test_mini_args import argument_parser


def one_hot_dynamic(labels_train, class_num=5):
    labels_train = labels_train.cpu()
    nKnovel = class_num
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot

def read_image(img_path):
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    if ('.png' in img_path) or ('.jpg' in img_path) or ('.bmp' in img_path):
        img = Image.open(img_path).convert('RGB')
        return img
    else:
        raise IOError("{} is not a image file".format(img_path))

def transform_image(img):
    transform = T.Compose([
        T.Resize((84, 84), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    return img


class SupportData(Dataset):
    def __init__(self, support_dir='/persons/jinxianglai/FewShotLearning/few-shot_classification/test_imgs/FSL_dataset/bottle'):
        # load file
        cat_container = sorted(os.listdir(support_dir))
        self.class_num = len(cat_container)
        cats2label = {cat:label for label, cat in enumerate(cat_container)}
        self.label2cats = {label:cat for label, cat in enumerate(cat_container)}
        self.support_dataset = []
        for cat in cat_container:
            for img_name in sorted(os.listdir(os.path.join(support_dir, cat))):
                label = cats2label[cat]
                img_path = os.path.join(support_dir, cat, img_name)
                self.support_dataset.append((img_path, label))

    def __len__(self):
        return len(self.support_dataset)

    def __getitem__(self, idx):
        img_path = self.support_dataset[idx][0]
        img = read_image(img_path)
        img = transform_image(img)
        support_data = img
        support_label = self.support_dataset[idx][1]
        return support_data, support_label

    def get_support_set(self):
        support_data_set = []
        support_label_set = []
        for idx in range(len(self.support_dataset)):
            support_data, support_label = self.__getitem__(idx)
            support_data_set.append(support_data)
            support_label_set.append(support_label)
        support_data_set = torch.stack(support_data_set, dim=0) # n, c, h, w
        support_label_set = torch.LongTensor(support_label_set) # n
        return support_data_set, support_label_set


def init_classifier():
    # loading configs
    parser = argument_parser()
    args = parser.parse_args()

    # define model
    model = Model(args=args)
    # load the model
    # checkpoint = torch.load(args.resume, map_location='cuda:0')
    checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu_devices[0]}')
    # checkpoint = torch.load(args.resume)
    multi_gpu_train = False
    for k, v in checkpoint['state_dict'].items():
        if 'module.' in k:
            multi_gpu_train = True
            break
    if multi_gpu_train:
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=False)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Loaded checkpoint from '{}'".format(args.resume))
    model = model.cuda()

    return model

def get_support_feats(model, support_data_set, support_label_set, class_num):
    # obtain support features
    support_data_set = support_data_set.cuda()
    support_label_set_1hot = one_hot_dynamic(support_label_set, class_num=class_num).cuda()
    _, support_feats, _, _ = model(support_data_set.unsqueeze(0), support_data_set.unsqueeze(0), support_label_set_1hot.unsqueeze(0), support_label_set_1hot.unsqueeze(0))
    return support_feats # 1, class_num, c, h, w

def inference_classifier(model, support_data_set, support_label_set, support_feats, class_num, label2cats, input_img):
    # support set
    support_data_set = support_data_set.cuda()
    support_label_set_1hot = one_hot_dynamic(support_label_set, class_num=class_num).cuda()
    # input img
    input_img = transform_image(input_img)
    input_img = input_img.cuda()
    input_img_1hot_tmp = support_label_set_1hot[0]

    model.eval()
    with torch.no_grad():
       output_results, _, _, _ = model(support_data_set.unsqueeze(0), input_img.unsqueeze(0).unsqueeze(0), support_label_set_1hot.unsqueeze(0), input_img_1hot_tmp.unsqueeze(0).unsqueeze(0), ftrain_in=support_feats)
       cls_scores = output_results['metric_scores']
       cls_scores = cls_scores.view(-1)
       _, pred = torch.max(cls_scores.detach().cpu(), 0)
       pred = pred.item()
    category = label2cats[pred]
    return pred, category

def inference_classifier_folder(model, support_data_set, support_label_set, support_feats, class_num, label2cats, input_img_folder, save_path=None):
    # read test imgs
    img_num = 0
    for img_name in sorted(os.listdir(input_img_folder)):
        # only process normal data
        # if '_normal.png' not in img_name:
        #     continue
        img_path = os.path.join(input_img_folder, img_name)
        print(img_path)
        img_num += 1
        print(f'{img_num} images are processed.')
        test_img = read_image(img_path)
        pred, category = inference_classifier(model, support_data_set, support_label_set, support_feats, class_num, label2cats, test_img)
        if save_path != None:
            img_save_path = save_path + '/' + str(category)
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            copy(img_path, img_save_path + '/' + img_name)

def accuracy_classifier_folder(model, support_data_set, support_label_set, support_feats, class_num, label2cats, input_img_folder, save_path=None):
    # read test imgs
    img_num = 0 # 图像总数
    pred_correct = 0 # 预测正确的数量
    cat_container = sorted(os.listdir(input_img_folder))
    cat_acc_dict = dict()
    cat_img_num_dict = dict()
    cat_pred_correct_dict = dict()
    for cat in cat_container:
        cat_img_num = 0
        cat_pred_correct = 0
        for img_name in sorted(os.listdir(os.path.join(input_img_folder, cat))):
            # only process normal data
            # if '_normal.png' not in img_name:
            #     continue
            img_path = os.path.join(input_img_folder, cat, img_name)
            print(img_path)
            img_num += 1
            cat_img_num += 1
            print(f'{img_num} images are processed.')
            test_img = read_image(img_path)
            pred, category = inference_classifier(model, support_data_set, support_label_set, support_feats, class_num, label2cats, test_img)
            if category == cat:
                pred_correct += 1
                cat_pred_correct += 1
            if save_path != None:
                img_save_path = save_path + '/' + str(category)
                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)
                copy(img_path, img_save_path + '/' + img_name)
        cat_acc_dict[cat] = 100.00 * float(cat_pred_correct) / float(cat_img_num)
        cat_pred_correct_dict[cat] = cat_pred_correct
        cat_img_num_dict[cat] = cat_img_num
    # accuracy
    acc = 100.00 * float(pred_correct) / float(img_num)
    # print(f'acc = {pred_correct}/{img_num} = {acc}%.')
    print('total acc = {}/{} = {:2f}%.'.format(pred_correct, img_num, acc))
    print("---------------------------------")
    for cat in cat_acc_dict.keys():
        print('{} acc = {}/{} = {:2f}%.'.format(cat, cat_pred_correct_dict[cat], cat_img_num_dict[cat], cat_acc_dict[cat]))
    print("---------------------------------")



# 推理：所有图像放在同个目录下
def inference_folder():
    # init model
    model = init_classifier()
    # get support dataset
    SD = SupportData(support_dir=os.path.join(sys.path[0], '/persons/jinxianglai/FewShotLearning/few-shot_classification/test_imgs/FSL_dataset/bottle_5shot'))
    support_data_set, support_label_set = SD.get_support_set()
    class_num = SD.class_num
    label2cats = SD.label2cats # {label:cat}
    # get_support_feats
    # support_feats = get_support_feats(model, support_data_set, support_label_set, class_num) # It needs to fix the bug!
    support_feats = None # looks ok, but it needs to further check!
    # inference
    input_img_folder = os.path.join(sys.path[0], '/persons/jinxianglai/FewShotLearning/few-shot_classification/test_imgs/FSL_dataset/bottle_5shot/broken_large')
    save_path = '/persons/jinxianglai/FewShotLearning/few-shot_classification/CAN_output/broken_large'
    inference_classifier_folder(model, support_data_set, support_label_set, support_feats, class_num, label2cats, input_img_folder, save_path=save_path)
    print('classifier done.')

# 计算accuracy：
def accuracy_folder():
    # init model
    model = init_classifier()
    # get support dataset
    SD = SupportData(support_dir=os.path.join(sys.path[0], '/persons/jinxianglai/FewShotLearning/few-shot_classification/test_imgs/FSL_dataset/bottle_5shot'))
    support_data_set, support_label_set = SD.get_support_set()
    class_num = SD.class_num
    label2cats = SD.label2cats # {label:cat}
    # get_support_feats
    # support_feats = get_support_feats(model, support_data_set, support_label_set, class_num) # It needs to fix the bug!
    support_feats = None # looks ok, but it needs to further check!
    # inference
    input_img_folder = os.path.join(sys.path[0], '/persons/jinxianglai/FewShotLearning/few-shot_classification/test_imgs/FSL_dataset/bottle')
    # save_path = '/persons/jinxianglai/FewShotLearning/few-shot_classification/CAN_output/bottle'
    save_path = None
    accuracy_classifier_folder(model, support_data_set, support_label_set, support_feats, class_num, label2cats, input_img_folder, save_path=save_path)
    print('classifier done.')


if __name__ == '__main__':
    # inference_folder()
    accuracy_folder()


