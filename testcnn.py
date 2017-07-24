import os
import cv2
import random
import numpy as np

def img2data(fname):
	print fname
	path = './pic/'
	img = cv2.imread(path + fname, 0)	

	border_left = []
	for col in range(0, 15):
		raw = [0] * 7
		border_left.append(raw)	
	

	border_right = []
	for col in range(0, 15):
		raw = [0] * 6
		border_right.append(raw)	
	

	border_top = []
	for col in range(0, 7):
		raw = [0] * 28
		border_top.append(raw)	

	border_bottom = []
	for col in range(0, 6):
		raw = [0] * 28
		border_bottom.append(raw)	

	img = np.hstack((img, np.array(border_right)))
	img = np.hstack((np.array(border_left), img))
	img = np.vstack((np.array(border_top), img))
	img = np.vstack((img, np.array(border_bottom)))	
	
	label = fname[0]
	img_id = fname
	img_data = img
	return img_id, img_data, label

def data_set():
	samples = {}
	p_samples = []
	n_samples = []
	
	for item in os.listdir('./pic'):
		imgId, img_data, label = img2data(item)
		sample = {}
		sample['data'] = img_data
		sample['label'] = label
		samples[imgId] = sample

		if label == '0':
			n_samples.append(imgId)
		else:
			p_samples.append(imgId)

	return samples, p_samples, n_samples

def split_samples(rate):
	samples, p_samples, n_samples = data_set()
	train_p_ids = random.sample(p_samples, int(rate * len(p_samples)))
	train_n_ids = random.sample(n_samples, int(rate * len(n_samples)))
	test_p_ids  = list(set(p_samples).difference(set(train_p_ids))) 
	test_n_ids  = list(set(n_samples).difference(set(train_n_ids))) 


	train_ids = list(set(train_p_ids) | set(train_n_ids))
	test_ids = list(set(test_p_ids)   | set(test_n_ids))

	return train_ids, test_ids, samples