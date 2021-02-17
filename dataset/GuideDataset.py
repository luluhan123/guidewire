#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-28 16:19:09
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch
import itertools
import numpy as np 

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class GuideDataset(Dataset):
	"""docstring for GuideDataset"""
	def __init__(self, txt_path, transforms=None, size=-1):
		super(GuideDataset, self).__init__()
		file_list = []
		try:
			fh = open(txt_path, 'r')
			for line in fh:
				word = line.rsplit()
				file_list.append(word[0])
		except IOError:
			print("Error: 没有找到txt文件或读取txt文件失败")
		self.file_list = file_list[:size]
		self.transforms = transforms

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		data = np.load(self.file_list[index])
		img, label = data['img'], data['label']

		img = img.astype(np.float32)
		img = np.expand_dims(img, 0)

		label = label.astype(np.int64)
		label = np.expand_dims(label, 0)

		sample = {'image': img, 'label': label}
		if self.transforms is not None:
			sample = self.transforms(sample)

		return sample
		

class TwoStreamBatchSampler(Sampler):
	"""docstring for TwoStreamBatchSampler"""
	def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
		self.primary_indices = primary_indices
		self.secondary_indices = secondary_indices
		self.secondary_batch_size = secondary_batch_size
		self.primary_batch_size = batch_size - self.secondary_batch_size

		assert len(self.primary_indices) >= self.primary_batch_size > 0
		assert len(self.secondary_indices) >= self.secondary_batch_size > 0
		
	def __iter__(self):
		primary_iter = iterate_once(self.primary_indices)
		secondary_iter = iterate_once(self.secondary_indices)
		return (
			primary_batch + secondary_batch
			for(primary_batch, secondary_batch)
			in zip(grouper(primary_iter, self.primary_batch_size), 
				grouper(secondary_iter, self.secondary_batch_size))
			)

	def __len__(self):
		return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
	return np.random.permutation(iterable)

def iterate_eternally(indices):
	def infinite_shuffles():
		yield np.random.permutation(indices)
	return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
	args = [iter(iterable)] * n
	return zip(*args)