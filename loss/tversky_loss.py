#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-14 10:21:06
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
	"""docstring for tversky_loss"""
	def __init__(self, beta=0.7):
		super(TverskyLoss, self).__init__()
		self.beta = beta
		self.alpha = 1.0 - self.beta

	def forward(self, output, target):
		N = target.size(0)

		output = torch.sigmoid(output)
		target = torch.sigmoid(target)
		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)

		tp = output_flat * target_flat
		fn = (1 - output_flat) * target_flat
		fp = output_flat * (1 - target_flat)

		loss = 1 - tp.sum(1) / (tp.sum(1) + self.alpha * fp.sum(1) + self.beta * fn.sum(1))
		loss = loss.sum() / N
		return loss 