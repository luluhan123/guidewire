#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-12-08 13:19:53
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import numpy as np
from scipy import ndimage

def mkdir(path):
    '''make dir'''

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def load_model(net, load_dir):
    checkpoint = torch.load(load_dir)
    net.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).cuda()
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        net = net.cuda()
    return net


def save_model(net, epoch, save_dir):
    '''save model'''

    mkdir(save_dir)

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    print(os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))
    torch.save({
        'epoch': epoch,
        'save_dir': save_dir,
        'state_dict': state_dict},
        os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_dis(img_gt, out_shape):
    img_gt = img_gt.astype(np.uint8)
    normalized_dis = np.zeros(out_shape)

    thershold_upper = 0.05
    thershold_lower = 0.03

    for b in range(out_shape[0]):
        mask = img_gt[b]
        mask.reshape(512, 512)
        if mask.any():
            distance_map = ndimage.distance_transform_edt(1 - mask)
            max_value = distance_map.max()
            min_value = distance_map.min()
            distance_upper = max_value * (1 - thershold_upper)
            distance_lower = max_value * thershold_lower

            distance_map[distance_map > distance_upper] = distance_upper
            distance_map[distance_map == 0] = distance_lower
            distance_map[distance_map < distance_lower] = distance_lower

            distance_map = (distance_map - min_value) / (max_value - min_value)

            normalized_dis[b] = distance_map
    return normalized_dis

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))