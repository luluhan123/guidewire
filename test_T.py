#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-06 15:01:37
# @Author  : lulu.han (lulu.han@aliyun.com)
# @Link    : https://github.com/luluhan123
# @Version : $Id$

import os
import cv2
import torch
import numpy as np
from model import unet
from utils import load_model
from data_prefetcher import data_prefetcher
from dataset import GuideWireDataset
from metric import *

BATCH_SIZE = 1
NUM_WORKERS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
	train_txt_path = './data/semi/train.txt'
	save_dir = './checkpoints/semi/test'
	T_save_path = './checkpoints/semi/T/model_at_epoch_486.dat'
	
	def worker_init_fn(worker_id):
		random.seed(seed + worker_id)

	model = unet(in_channel=1, n_classes=1)
	model = load_model(model, T_save_path)
	model = model.cuda()
	model.eval()

	train_dataset = GuideDataset(train_txt_path, size=114)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)
	prefetcher = data_prefetcher(test_loader)
	input, target= prefetcher.next()

	i = -1
	while input is not None:
		# target = torch.nn.functional.one_hot(target, 2).reshape(target.size(0), 2, 512, 512)

		with torch.no_grad():
			gt_dis = compute_dis(target[:].cpu().numpy(), target[:].shape)
			gt_dis = torch.from_numpy(gt_dis).float().cuda()

		output = model(gt_dis)
		output = torch.sigmoid(output).squeeze().data.cpu().numpy()
		output[output < 0.5] = 0
		output[output >= 0.5] = 1

		cv2.imwrite(os.path.join(save_dir, str(i) + '_output.jpg'), output * 255)
		print(str(i) + ' finish!')