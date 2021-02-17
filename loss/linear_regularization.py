#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-13 11:35:31
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class linear_regularization(nn.Module):
	"""docstring for linear_regularization"""
	def __init__(self, alpha=0.01):
		super(linear_regularization, self).__init__()
		# self.kernel = [[0.1, 0.1, 0.1], [0.1, -0.8, 0.1], [0.1, 0.1, 0.1]]
		# self.kernel = [[0.0751, 0.1238, 0.0751], [0.1238, -0.7958, 0.1238], [0.0751, 0.1238, 0.0751]]
		self.kernel = [[0.1414, 0.1, 0.1414], [0.1, -0.9656, 0.1], [0.1414, 0.1, 0.1414]]
		self.alpha = alpha

	def forward(self, output):
		output = torch.sigmoid(output)
		channel = output.size()[1]
		kernel = torch.FloatTensor(self.kernel).expand(channel, channel, 3, 3)
		weight = nn.Parameter(data=kernel, requires_grad=False).cuda()
		conv = F.conv2d(output, weight, padding=1)
		return self.alpha * torch.mean(conv)