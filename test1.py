#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-17 11:48:23
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import cv2
import torch
import numpy as np
from model import unet
from utils import load_model
from data_prefetcher import data_prefetcher
from dataset import GuideWireDataset

BATCH_SIZE = 1
NUM_WORKERS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
	model_dir = './checkpoints/seg2/DismapDiceLoss_k3/model_at_epoch_042.dat'
	save_dir = './test/1015/DismapDiceLoss_k3'
	test_txt_path = './data/seg1/test.txt'

	model = unet(in_channel=1, n_classes=1)
	model = load_model(model, model_dir)
	model = model.cuda()
	model.eval()

	test_dataset = GuideWireDataset(test_txt_path)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	prefetcher = data_prefetcher(test_loader)
	input, _, _ = prefetcher.next()

	i = -1
	while input is not None:
		i += 1
		with torch.no_grad():
			output = model(input)

			output = torch.sigmoid(output).squeeze().data.cpu().numpy()
			output[output < 0.5] = 0
			output[output >= 0.5] = 1
			cv2.imwrite(os.path.join(save_dir, str(i) + '_output.jpg'), output * 255)
			print(str(i) + ' finish!')

		input, _, _ = prefetcher.next()

if __name__ == '__main__':
	main()