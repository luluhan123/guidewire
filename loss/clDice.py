#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-21 11:35:31
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class clDice(nn.Module):
	"""docstring for clDice"""
	def __init__(self):
		super(clDice, self).__init__()

	def forward(self, output, target):
		N = target.size(0)
		output = torch.sigmoid(output)
		smooth = 1
		alpha = 0.7

		clDice = 0
		for i in range(N):
			img = output[i, :, :]
			tar = target[i, :, :]
			clDice += (1 - soft_clDice(img, tar))

		clDice = clDice.sum() / N

		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)
		intersection = output_flat * target_flat
		Dice = 1 - 2 * (intersection.sum(1) + smooth) / (output_flat.sum(1) + target_flat.sum(1) + smooth)
		Dice = Dice.sum() / N

		loss = alpha * Dice + (1 - alpha) * clDice
		return loss

def soft_erode(img):
	p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
	p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
	return torch.min(p1, p2)

def soft_dilate(img):
	return F.max_pool2d(img, (3, 3), (1, 1), (1 , 1))

def soft_open(img):
	return soft_dilate(soft_erode(img))

def soft_skel(img, iter):
	img1 = soft_open(img)
	skel = F.relu(img - img1)
	for j in range(iter):
		img = soft_erode(img)
		img1 = soft_open(img)
		delta = F.relu(img - img1)
		skel = skel + F.relu(delta - skel * delta)
	return skel

def soft_clDice(v_p, v_l, iter=50, smooth=1):
	s_p = soft_skel(v_p, iter)
	s_l = soft_skel(v_l, iter)
	tprec = ((s_p * v_l).sum() + smooth) / (s_p.sum() + smooth)
	tsens = ((s_l * v_p).sum() + smooth) / (s_l.sum() + smooth)
	return (2 * tprec * tsens) / (tprec + tsens)