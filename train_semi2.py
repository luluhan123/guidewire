#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-29 22:21:51
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

from model import SemiUnet, unet
from utils import *
from loss import *
from metric import * 
from datetime import datetime
from dataset import GuideDataset, TwoStreamBatchSampler
from data_prefetcher import data_prefetcher
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss, MSELoss

EPOCHS = 500
BATCH_SIZE = 4
LABEL_BS = 2
NUM_WORKERS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

num_classes = 2
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

consistency = 0.1
consistency_rampup = 30.0

result_path = '.Result/semi/FCDDis'
save_dir = './checkpoints/semi/FCDDis'
T_save_path = './checkpoints/semi/T/model_at_epoch_486.dat'

def get_current_consistency_weight(epoch):
	return consistency * sigmoid_rampup(epoch, consistency_rampup)

if __name__ == '__main__':
	train_txt_path = './data/semi/train.txt'
	train_dataset = GuideDataset(train_txt_path)

	labelnum = 1140
	labeled_idx = list(range(labelnum))
	unlabeled_idx = list(range(labelnum, 5650))
	batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, BATCH_SIZE, BATCH_SIZE - LABEL_BS)

	def worker_init_fn(worker_id):
		random.seed(seed + worker_id)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)

	model = SemiUnet(n_classes=1)
	model = model.cuda()
	T = unet(in_channel=1, n_classes=1)
	T = load_model(T, T_save_path)
	T = T.cuda()

	ce_loss = BCEWithLogitsLoss()
	mse_loss = MSELoss()
	criterion = DiceLoss()
	regularization = linear_regularization(alpha=0.1)

	optimizer = optim.Adam(model.parameters(), lr = 0.00001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

	start_epoch = 1
	now_time = datetime.now()
	time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
	log_dir = os.path.join(result_path, time_str)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	writer = SummaryWriter(log_dir=log_dir)
	print('Start Train!')

	for epoch in range(start_epoch, EPOCHS):
		batch_time = AverageMeter()
		lossSegMeter = AverageMeter()
		lossDisMeter = AverageMeter()
		lossMeter = AverageMeter()
		supervisedLossMeter = AverageMeter()
		DLossMeter = AverageMeter()

		prefetcher = data_prefetcher(train_loader)
		input, target= prefetcher.next()
		i = 0

		end = time.time()
		if epoch < 6:
			beta = 0.05 * epoch
		else:
			beta = 0.3

		while input is not None:
			i += 1
			input = input.cuda()
			target = target.cuda()

			# Generate Discriminator target based on sampler
			model.train()
			T.eval()

			outputs_seg, outputs_dis = model(input)

			# calculate the loss
			with torch.no_grad():
				gt_dis = compute_dis(target[:].cpu().numpy(), outputs_dis[:LABEL_BS, ...].shape)
				gt_dis = torch.from_numpy(gt_dis).float().cuda()

			loss_dis = mse_loss(outputs_dis[:LABEL_BS, ...], gt_dis)
			# loss_seg = criterion(outputs_dis[:LABEL_BS, ...], target[:LABEL_BS, ...], gt_dis)
			loss_seg = criterion(outputs_seg[:LABEL_BS, ...], target[:LABEL_BS, ...])

			consistency_weight = get_current_consistency_weight(epoch)
			supervised_loss = loss_seg  + beta * loss_dis

			Toutputs = T(outputs_dis)
			loss_adv = criterion(Toutputs, outputs_seg)
			loss = supervised_loss  + consistency_weight * loss_adv

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 2 == 0:	
				batch_time.update(time.time() - end)
				lossSegMeter.update(loss_seg.item(), input.size(0))
				lossDisMeter.update(loss_dis.item(), input.size(0))
				lossMeter.update(loss.item(), input.size(0))
				supervisedLossMeter.update(supervised_loss.item(), input.size(0))
				DLossMeter.update(loss_adv.item(), input.size(0))

			input, target= prefetcher.next()

			if((i + 1) % 10 == 0):
				print('Train Epoch: {}\t [{}/{} ({:.0f}%)]\t Time: {:.3f} ({:.3f}) \
					Loss: {:.6f} ({:.6f})\t Loss_dis: {:.6f} ({:.6f})\t \
					Loss_seg: {:.6f} ({:.6f})\t supervised_loss: {:.6f} ({:.6f})\t \
					adv_loss: {:.6f} ({:.6f})\t consistency_weight: {:6f} \t bate: {:.6f}'.format(\
						epoch, i * len(input), len(batch_sampler), \
						100. * i / len(train_loader), batch_time.val, batch_time.avg, lossMeter.val, \
						lossMeter.avg, lossDisMeter.val, lossDisMeter.avg, lossSegMeter.val, \
						lossSegMeter.avg, supervisedLossMeter.val, supervisedLossMeter.avg, \
						DLossMeter.val, DLossMeter.avg, consistency_weight, beta))

				writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
				writer.add_scalar('loss/loss', loss, epoch)
				writer.add_scalar('loss/loss_dis', loss_dis, epoch)
				writer.add_scalar('loss/loss_seg', loss_seg, epoch)
				writer.add_scalar('train/consistency_weight', consistency_weight, epoch)
				writer.add_scalar('loss/loss_adv', consistency_weight * loss_adv, epoch)
				writer.add_scalar('GAN/loss_adv', loss_adv,epoch)

		scheduler.step()
		save_model(model, epoch, save_dir)
