import cv2
import numpy as np 
import copy
import os
import cv

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
	def __init__(self, arg):
		super(Dataset, self).__init__()
		self.arg = arg
		

	def img2matrix(fpath):
	 	flist = os.listdir(fpath)
	 	for fname in flist:
	 		pass 
