#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-12-07 16:19:09
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import numpy as np

from torch.utils.data import Dataset


class GuideWireDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        super(GuideWireDataset, self).__init__()
        file_list = []
        try:
            fh = open(txt_path, 'r')
            for line in fh:
                word = line.rsplit()
                file_list.append(word[0])
        except IOError:
            print("Error: 没有找到txt文件或读取txt文件失败")

        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        try:
            data = np.load(self.file_list[index])
            img, label, distance = data['img'], data['label'], data['distance']

            img = img.astype(np.float32)
            img = np.expand_dims(img, 0)

            label = label.astype(np.float32)
            label = np.expand_dims(label, 0)

            distance = distance.astype(np.float32)
            distance = np.expand_dims(distance, 0)

            if self.transform is not None:
                catTensor = np.concatenate((img, label), axis=0)
                catTensor = self.transform(catTensor)
                img, label = catTensor.split([1, 1], dim=0)
                print(img.size(), label.size())
            return img, label, distance
        except IOError:
            print("Error: 没有找到文件或读取文件失败, ", self.file_list[index])
