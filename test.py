#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-17 11:48:23
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import cv2
import torch
import numpy as np
from model import unet, SegNet
from utils import load_model
from data_prefetcher import data_prefetcher
from dataset import GuideWireDataset
from metric import *

BATCH_SIZE = 1
NUM_WORKERS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
	model_dir = './checkpoints/seg2/segnet_gen1/model_at_epoch_013.dat'
	save_dir = './test/0610/segnet_gen1/test'
	test_txt_path = './data/seg/valid.txt'

	# model = unet(in_channel=1, n_classes=1)
	model = SegNet(input_nbr = 1, label_nbr = 1)
	model = load_model(model, model_dir)
	model = model.cuda()
	model.eval()

	test_dataset = GuideWireDataset(test_txt_path)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	prefetcher = data_prefetcher(test_loader)
	input, target, distance = prefetcher.next()

	dice = []
	IoU = []
	precision = []
	recall = []

	i = -1
	while input is not None:
		i += 1
		with torch.no_grad():
			output = model(input)
			dice.append(dice_coeff(output, target).item())
			IoU.append(iou_coeff(output, target).item())
			precision.append(Precision(output, target).item())
			recall.append(Recall(output, target).item())

			output = torch.sigmoid(output).squeeze().data.cpu().numpy()
			output[output < 0.5] = 0
			output[output >= 0.5] = 1
			# output = torch.argmax(output, dim=1).squeeze().data.cpu().numpy()
			# output = output.squeeze().data.cpu().numpy()
			# output = np.argmax(output, axis=0)
			cv2.imwrite(os.path.join(save_dir, str(i) + '_output.jpg'), output * 255)
			print(str(i) + ' finish!')

		input, target, distance = prefetcher.next()
	print('dice: ', np.mean(dice), np.max(dice), np.min(dice), np.std(dice))
	print('iou: ', np.mean(IoU), np.max(IoU), np.min(IoU), np.std(IoU))
	print('precision: ', np.mean(precision), np.max(precision), np.min(precision), np.std(precision))
	print('recall: ', np.mean(recall), np.max(recall), np.min(recall), np.std(recall))

if __name__ == '__main__':
	main()