import os
import cv2
import random
import numpy as np
from feature import *

def img2data(fname):
	path = './pic/'
	img = cv2.imread(path + fname, 0)
	print img.shape

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
	img_data = []
	hot_code = { 
		'0':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		'1':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	}

	for i in range(0, 28):
		for j in range(0, 28):
			img_data.append(img[i][j])

	img_data = np.array(img_data)
	label = hot_code[label]
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

def get_img_data(fname):
	path = './pic/'
	p_samples = []
	n_samples = []
	img = cv2.imread(path + fname, 0)
	img = img.reshape(1, 225)[0]
	label = fname[0]
	return img, label, fname

def get_data_set():
	samples = {}
	n_samples = []
	p_samples = []

	for item in os.listdir('./pic/'):
		img, label, fname = get_img_data(item)

		if label == '0':
			n_samples.append(fname)
		else:
			p_samples.append(fname)
		
		hot_code = { 
			'0':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			'1':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
		}
		
		sample = {}
		sample['data'] = img
		sample['label'] = hot_code[label]
		samples[fname] = sample

	return samples, n_samples, p_samples

def split_data_set(rate):
	samples, p_samples, n_samples = get_data_set()
	train_p_ids = random.sample(p_samples, int(rate * len(p_samples)))
	train_n_ids = random.sample(n_samples, int(rate * len(n_samples)))
	test_p_ids  = list(set(p_samples).difference(set(train_p_ids))) 
	test_n_ids  = list(set(n_samples).difference(set(train_n_ids))) 

	print len(train_p_ids)
	print len(train_n_ids)
	print len(test_n_ids)
	print len(train_n_ids)

	train_ids = list(set(train_p_ids) | set(train_n_ids))
	test_ids = list(set(test_p_ids)   | set(test_n_ids))

	return train_ids, test_ids, samples

def load_data(rate):
	train_data = []
	train_label = []
	test_data = []
	test_label = []
	train_ids, test_ids, samples = split_data_set(0.8)
	
	for Id in  train_ids:
		train_data.append(list(samples[Id]['data']))
		train_label.append(samples[Id]['label'])
	
	for Id in  test_ids:
		test_data.append(list(samples[Id]['data']))
		test_label.append(samples[Id]['label'])

	return train_data, train_label, test_data, test_label

if __name__ == "__main__":
	'''
	train_data, train_label, test_data, test_label = load_data(0.8)
	
	trainLabel = [] 
	for label in train_label:
		if label[0] == 1:
			trainLabel.append(0)
		else:
			trainLabel.append(1)
	trainLabel = np.array(trainLabel)
	
	for label in test_label:

		label = np.array(label)
	test_label = np.array(test_label)

	print train_label
	classify(list(train_data), trainLabel)
	'''
	pass