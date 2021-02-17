#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-11-15 18:02:28
# @Author  : lulu.han (lulu.han@aliyun.com)
# @Link    : https://github.com/luluhan123
# @Version : $Id$

import os
import time
import torch
import random
import numpy as np

from dataset import GuideDataset, TwoStreamBatchSampler
from data_prefetcher import data_prefetcher

EPOCHS = 500
BATCH_SIZE = 4
LABEL_BS = 2
NUM_WORKERS = 1

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__ == '__main__':
	train_txt_path = './data/semi/train.txt'
	train_dataset = GuideDataset(train_txt_path)
	labelnum = 1140
	labeled_idx = list(range(labelnum))
	unlabeled_idx = list(range(labelnum, 1500))
	batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, BATCH_SIZE, BATCH_SIZE - LABEL_BS)

	def worker_init_fn(worker_id):
		random.seed(seed + worker_id)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)
	prefetcher = data_prefetcher(train_loader)

	input, target= prefetcher.next()
	i = 0
	while input is not None:
		i += 1
		print(i, input.shape, target.shape)
		input, target= prefetcher.next()