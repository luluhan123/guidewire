#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-16 22:21:51
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import unet, SegNet, FCN8s
from utils import *
from loss import *
from metric import * 
from datetime import datetime
from dataset import GuideWireDataset
from data_prefetcher import data_prefetcher
from tensorboardX import SummaryWriter

EPOCHS = 500
BATCH_SIZE = 4
NUM_WORKERS = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

result_path = './Result/seg2/FCN_GenDice'
save_dir = './checkpoints/seg2/FCN_GenDice'

def train(train_loader, model, criterion, regularization, optimizer, scheduler, writer, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	dice = AverageMeter()
	iou = AverageMeter()

	model.train()
	end = time.time()

	prefetcher = data_prefetcher(train_loader)
	input, target, distance = prefetcher.next()
	i = 0
	while input is not None:
		i += 1
		input = input.cuda()
		target = target.cuda()
		distance = distance.cuda()

		output = model(input)
		# print(output.shape, target.shape)

		# loss = criterion(output, target, distance) + regularization(output)
		loss = criterion(output, target, type='sigmoid')
		dice_coef = dice_coeff(output, target)
		iou_coef = iou_coeff(output, target)

		losses.update(loss.item(), input.size(0))
		dice.update(dice_coef.item(), input.size(0))
		iou.update(iou_coef.item(), input.size(0))
		batch_time.update(time.time() - end)
		end = time.time()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if(i + 1) % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTime: {:.3f} ({:.3f})\tLoss: {:.6f} ({:.6f})\tDice Coef: {:.6f}({:.6f})\tIoU: {:.6f}({:.6f})'.\
				format(epoch, i * len(input), len(train_loader.dataset), 100. * i / len(train_loader), batch_time.val, batch_time.avg, losses.val, losses.avg, dice.val, dice.avg, iou.val, iou.avg))
				# 记录训练loss
			writer.add_scalars('Loss_group', {'train_loss': losses.avg}, epoch)
			writer.add_scalars('dice_group', {'train_dice': dice.avg}, epoch)
			writer.add_scalars('iou_group', {'train_iou': iou.avg}, epoch)
			writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)

		input, target, distance = prefetcher.next()
	scheduler.step()
	print('Train Epoch: {}\t Loss: {:.6f}\t Dice: {:.6f}\t IoU: {:.6f}'.format(epoch, losses.avg, dice.avg, iou.avg))

def valid(valid_loader, model, criterion, regularization, writer, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	dice = AverageMeter()
	iou = AverageMeter()

	model.eval()
	end = time.time()

	prefetcher = data_prefetcher(valid_loader)
	input, target, distance = prefetcher.next()

	i = 0
	while input is not None:
		i += 1
		input = input.cuda()
		target = target.cuda()
		distance = distance.cuda()

		with torch.no_grad():
			output = model(input)
			# loss = criterion(output, target, distance) + regularization(output)
			loss = criterion(output, target, type='sigmoid')
			dice_coef = dice_coeff(output, target)
			iou_coef = iou_coeff(output, target)

		losses.update(loss.item(), input.size(0))
		dice.update(dice_coef.item(), input.size(0))
		iou.update(iou_coef.item(), input.size(0))
		batch_time.update(time.time() - end)
		end = time.time()

		input, target, distance = prefetcher.next()

	writer.add_scalars('Loss_group', {'valid_loss': losses.avg}, epoch)
	writer.add_scalars('dice_group', {'valid_dice': dice.avg}, epoch)
	writer.add_scalars('iou_group', {'valid_iou': iou.avg}, epoch)
	print('valid Epoch: {}\tLoss: {:.6f} Dice: {:.6f} IoU: {:.6f}'.format(epoch, losses.avg, dice.avg, iou.avg))

def main():
	train_txt_path = './data/seg/train.txt'
	valid_txt_path = './data/seg/valid.txt'

	train_dataset = GuideWireDataset(train_txt_path)
	valid_dataset = GuideWireDataset(valid_txt_path)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	# model = unet(in_channel=1, n_classes=1)
	# model = SegNet(input_nbr = 1, label_nbr = 1)
	model = FCN8s(n_class=1)
	model = model.cuda()
	# model = nn.DataParallel(model, device_ids=[1, 3])

	# criterion = ComboLoss(alpha=0.5, beta=0.4)
	# criterion = DismapDiceLoss()
	# criterion = DiceLoss()
	# criterion = clDice()
	# criterion = FocalLoss2d()
	# criterion = DismapCE()
	# criterion = DismapComboLoss(alpha=0.9)
	# criterion = clDice()
	# criterion = TverskyLoss()
	# criterion = IoULoss()
	criterion = GenDiceLoss()
	regularization = linear_regularization(alpha=0.1)
	optimizer = optim.Adam(model.parameters(), lr=0.00001)
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
		train(train_loader, model, criterion, regularization, optimizer, scheduler, writer, epoch)
		valid(valid_loader, model, criterion, regularization, writer, epoch)
		save_model(model, epoch, save_dir)


if __name__ == '__main__':
	main()