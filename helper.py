import cv2
import numpy as np 
import copy
import os
import cv2

def resizeImg(fname):
	image=cv2.imread(fname)	

	height = image.shape[0]
	width  = image.shape[1]
	delta  = abs(width - height)	

	if width > height:
		blank = np.array([255] * delta * width * 3).reshape(delta, width, 3)
		image = np.vstack((image, blank))
	else:
		blank = np.array([255] * delta * height * 3).reshape(height, delta, 3)
		image = np.hstack((image, blank))	

	cv2.imwrite('blank.jpg', image)
	image = cv2.imread('blank.jpg')
	os.system('rm blank.jpg')
	res=cv2.resize(image, (227, 227),interpolation=cv2.INTER_CUBIC)
	cv2.imwrite(fname, res)


class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self):
		super(Dataset, self).__init__()
		self.labels = []
		self.data = []
		self.generate

	def generate(self):
		self.img2matrix('pic')
		
	def img2matrix(self, fpath, rate):
	 	flist = os.listdir(fpath)

	 	data = []
	 	labels = []
	 	for fname in flist:
	 		img = cv2.imread(fpath + '/'+ fname, 0)
	 		res=cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
	 		img = list(res.reshape(784))
	 		#print len(img)
	 		img = np.asarray(img)
	 		self.data.append(img)
	 		if fname[0] == 0:
	 			self.labels.append(np.asarray([1,0,0,0,0,0,0,0,0,0]))
	 		else:
	 			self.labels.append(np.asarray([0,1,0,0,0,0,0,0,0,0]))

	 	self.data = np.array(self.data)
	 	self.labels = np.array(self.labels)

	 	trainSize = int(len(self.data) * rate)
	 	self.traindata = self.data[ :trainSize]
	 	self.trainlabels = self.labels[ :trainSize]
	 	self.testdata = self.data[trainSize: ]
	 	self.testlabels = self.labels[trainSize: ]
	 	return self.data, self.labels

	def next_batch(self, num):
	 	idx = np.arange(0, len(self.traindata))
	 	np.random.shuffle(idx)
	 	print idx
	 	idx = idx[: num]
	 	data_shuffle = [self.traindata[i] for i in idx ]
	 	labels_shuffle = [self.trainlabels[i] for i in idx]

	 	return np.asarray(data_shuffle), np.asarray(labels_shuffle)
'''
if __name__ == "__main__":
	ds = Dataset()
	ds.img2matrix('pic')
	batch_xs, batch_ys =  ds.next_batch(100)
	print batch_xs.shape
	print batch_xs.shape
'''
