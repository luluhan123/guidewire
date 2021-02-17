#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-12-14 11:35:31
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenDiceLoss(nn.Module):
    """docstring for DiceLoss"""

    def __init__(self):
        super(GenDiceLoss, self).__init__()

    def forward(self, output, target, type='sigmoid'):
        N = target.size(0)
        C = target.size(1)
        smooth = 1

        if type == 'sigmoid':
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, 1)

        intersection = torch.tensor(0).float().cuda()
        union = torch.tensor(0).float().cuda()

        for i in range(C):
            w = 1 / (target[:, i].sum().pow(2))
            output_flat = output[:, i].view(N, -1)
            target_flat = target[:, i].view(N, -1)
            intersection += w * (output_flat * target_flat).sum()
            union += w * (output_flat + target_flat).sum()

        loss = 1 - 2 * (intersection / union)
        loss = loss / N
        return loss
