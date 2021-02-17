#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-28 20:58:41
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCDiscriminator(nn.Module):
	"""docstring for FCDiscriminator"""
	def __init__(self, num_classes=2, ndf=64, n_channel=1):
		super(FCDiscriminator, self).__init__()
		self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=8, stride=4, padding=2)
		self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=8, stride=4, padding=2)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=8, stride=4, padding=2)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Linear(ndf*4, 2)
		self.avgpool = nn.AvgPool2d((7, 7))

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.dropout = nn.Dropout2d(0.5)

	def forward(self, map, feature):
		map_feature = self.conv0(map)
		image_feature = self.conv1(feature)
		x = torch.add(map_feature, image_feature)

		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.dropout(x)

		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.dropout(x)

		x = self.conv4(x)
		x = self.leaky_relu(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x