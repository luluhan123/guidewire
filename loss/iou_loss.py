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


class IoULoss(nn.Module):
    """docstring for IoULoss"""

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, output, target, type='sigmoid'):
        N = target.size(0)
        smooth = 1

        if type == 'sigmoid':
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, 1)
        output_flat = output.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = output_flat * target_flat
        union = output_flat + target_flat - intersection

        loss = 1 - 2 * (intersection.sum(1) + smooth) / (union.sum(1) + smooth)
        loss = loss.sum() / N

        return loss
