import os
import cv2
import numpy as np
from glob import glob
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from medpy import metric

path = 'D:\\Han\\Guidewire\\test\\0610\\segnet_gen1\\test'
label_path = 'D:\\Han\\Guidewire\\test\\0610\\png'

file_list = glob(os.path.join(path, '*.jpg'))
label_list = glob(os.path.join(label_path, '*.png'))

ratio = []
lenacc = []
hd = []

for i in range(len(file_list)):
	img = cv2.imread(file_list[i], 0)
	img = img > 128
	labels = label(img, connectivity=2)
	props = regionprops(labels)
	eular = 0
	for j in range(0, labels.max()):
		eular += props[j].euler_number

	label_ = cv2.imread(label_list[i], 0)
	label_ = label_ > 128
	label_labels = label(label_, connectivity=2)
	label_props = regionprops(label_labels)
	label_eular = 0
	for j in range(0, label_labels.max()):
		label_eular += label_props[j].euler_number

	ratio.append(eular / label_eular)

	skeleton_img = skeletonize(img)
	skeleton_label = skeletonize(label_)
	lenacc.append(np.sum(skeleton_img) / np.sum(skeleton_label))

	hd.append(metric.binary.hd95(img, label_))
	print(file_list[i])


print(np.mean(ratio), np.max(ratio), np.min(ratio), np.std(ratio))
print(np.mean(lenacc), np.max(lenacc), np.min(lenacc), np.std(lenacc))
print(np.mean(hd), np.max(hd), np.min(hd), np.std(hd))