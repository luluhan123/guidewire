#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-07 11:35:31
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComboLoss(nn.Module):
	"""docstring for ComboLoss"""
	def __init__(self, alpha, beta):
		super(ComboLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta

	def forward(self, output, target):
		N = target.size(0)
		smooth = 1

		output = torch.sigmoid(output)
		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)

		intersection = output_flat * target_flat
		dice =  2 * (intersection.sum(1) + smooth) / (output_flat.sum(1) + target_flat.sum(1) + smooth)

		FN = self.beta * target_flat * output_flat.log()
		FP = (1 - self.beta) * (1 - target_flat) * (1 - output_flat).log()
		ce = (FN.sum(1) + FP.sum(1)) / output_flat.size(1)
		combo_loss =  - self.alpha * ce - (1 - self.alpha) * dice
		combo_loss = combo_loss.sum() / N
		return combo_loss 
