#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-11-27 22:21:51
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from model import unet
from utils import *
from loss import *
from metric import * 
from datetime import datetime
from dataset import GuideDataset
from data_prefetcher import data_prefetcher
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss, MSELoss

EPOCHS = 500
BATCH_SIZE = 8
NUM_WORKERS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

num_classes = 1
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__ == '__main__':
	train_txt_path = './data/semi/train.txt'
	save_dir = './checkpoints/T/0107/'
	train_dataset = GuideDataset(train_txt_path, size=1140)
	print(len(train_dataset))

	def worker_init_fn(worker_id):
		random.seed(seed + worker_id)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)

	model = unet(in_channel=1, n_classes=1)
	model = model.cuda()
	# mse_loss = MSELoss()
	criterion = DiceLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.00001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

	start_epoch = 1

	for epoch in range(start_epoch, EPOCHS):
		prefetcher = data_prefetcher(train_loader)
		losses = AverageMeter()
		dice = AverageMeter()
		input, target= prefetcher.next()
		i = 0

		while target is not None:
			i += 1
			target = target.cuda()
			# target = torch.nn.functional.one_hot(target, 2).reshape(target.size(0), 2, 512, 512)

			with torch.no_grad():
				gt_dis = compute_dis(target[:].cpu().numpy(), target[:].shape)
				gt_dis = torch.from_numpy(gt_dis).float().cuda()

			output = model(gt_dis)

			loss = criterion(output, target)

			losses.update(loss.item(), target.size(0))

			dice_coef = dice_coeff(output, target)
			dice.update(dice_coef.item(), input.size(0))
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if(i + 1) % 10 == 0:
				print('Train Epoch: {}[{}/{} ({:.0f}%)]\tloss: {:.6f} ({:.6f})\tdice: {:.6f} ({:.6f})'.format(epoch, i * len(target), len(train_loader.dataset), 100. * i / len(train_loader), losses.val, losses.avg, dice.val, dice.avg))

			input, target= prefetcher.next()
		scheduler.step()

		save_model(model, epoch, save_dir)