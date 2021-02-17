#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-16 11:35:31
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DismapDiceLoss(nn.Module):
	"""docstring for DismapDiceLoss"""
	def __init__(self):
		super(DismapDiceLoss, self).__init__()

	def forward(self, output, target, distance_map, type='sigmoid'):
		N = target.size(0)
		smooth = 1e-4

		if type == 'sigmoid':
			output = torch.sigmoid(output)
		else:
			output = torch.softmax(output, 1)
		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)
		distance_map_flat = distance_map.view(N, -1)

		intersection = output_flat * target_flat
		union = output_flat + target_flat
		loss = 1 - 2 * (intersection.sum(1) + smooth) / (((distance_map_flat + 1) * union).sum(1) - (distance_map_flat * intersection).sum(1))
		loss = loss.sum() / N
		return loss

class DismapCE(nn.Module):
	"""docstring for DismapCE"""
	def __init__(self):
		super(DismapCE, self).__init__()
		
	def forward(self, output, target, distance_map):
		smooth = 1e-4
		N = target.size(0)
		if type == 'sigmoid':
			output = torch.sigmoid(output)
		else:
			output = torch.softmax(output, 1)
		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)

		FN = target_flat * (output_flat + smooth).log()
		FP = (1 - target_flat) * (1 - output_flat + smooth).log()

		distance_map_flat = distance_map.view(N, -1)

		loss = ((FN + FP) * (distance_map_flat + 1)).sum(1) / output_flat.size(1)
		loss = -loss.sum() / N
		return loss

class DismapComboLoss(nn.Module):
	"""docstring for DismapComboLoss"""
	def __init__(self, alpha=0.5):
		super(DismapComboLoss, self).__init__()
		self.alpha = alpha

	def forward(self, output, target, distance_map, type='sigmoid'):
		smooth = 1
		N = target.size(0)
		if type == 'sigmoid':
			output = torch.sigmoid(output)
		else:
			output = torch.softmax(output, 1)
		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)
		distance_map_flat = distance_map.view(N, -1)

		intersection = output_flat * target_flat
		union = output_flat + target_flat

		dice = 2 * (intersection.sum(1) + smooth) / (((distance_map_flat + 1) * union).sum(1) - (distance_map_flat * intersection).sum(1))

		FN = target_flat * output_flat.log()
		FP = (1 - target_flat) * (1 - output_flat).log()
		ce = ((FN + FP) * (distance_map_flat + 1)).sum(1) / output_flat.size(1)

		combo_loss =  - self.alpha * ce - (1 - self.alpha) * dice
		combo_loss = combo_loss.sum() / N
		return combo_loss