#coding = utf-8
import math
import numpy as np
import logging
from matplotlib import pyplot as plt
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import svm, datasets  
from sklearn.metrics import roc_curve, auc  
from sklearn.cross_validation import StratifiedKFold 
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings("ignore")

def calcuateSpeed(sample, sid):
	if sample[-2] - sample[2] == 0:
		timeCost = 0.00001
	else:
		timeCost = (sample[-2] - sample[2])

	xspeed = (sample[-4] - sample[0]) / timeCost 
	yspeed = (sample[-3] - sample[1]) / timeCost
	distace = math.sqrt((sample[-4] - sample[0])**2 + (sample[-3] - sample[1])**2)
	speed = round (distace / timeCost, 6)

	xcur = 0
	ycur = 1 
	tcur = 2 
	avgxSpeed = 0
	avgySpeed = 0
	avgSpeed = 0
	count = 1
	while tcur < len(sample) - 3:
		delatX = sample[xcur+3] - sample[xcur]
		delatY = sample[ycur+3] - sample[ycur]
		delatT = sample[tcur+3] - sample[tcur]
		xcur += 3
		ycur += 3
		tcur += 3
		if delatT == 0:
			delatT = 0.001

		avgxSpeed += delatX / delatT
		avgySpeed += delatY / delatT
		avgSpeed  += math.sqrt(delatX**2 +  delatY**2) / delatT
		count += 1
	
	avgxSpeed = avgxSpeed / count
	avgySpeed = avgySpeed / count
	avgSpeed = avgSpeed / count

	return [xspeed, yspeed, distace, speed, avgxSpeed, avgySpeed, avgSpeed]

def calcuateAcculation(sample, sid):
	if sample[-2] - sample[2] == 0:
		timeCost = 0.00001
	else:
		timeCost = (sample[-2] - sample[2])

	xAccelation = (sample[-4] - sample[0]) / timeCost**2 
	yAccelation = (sample[-3] - sample[1]) / timeCost**2 
	distace = math.sqrt((sample[-4] - sample[0])**2 + (sample[-3] - sample[1])**2)
	Accelation = round (distace / timeCost**2 , 6)

	xcur = 0
	ycur = 1 
	tcur = 2 
	avgxAcc= 0
	avgyAcc = 0
	avgAcc = 0
	count = 1

	while tcur < len(sample) - 3:
		delatX = sample[xcur+3] - sample[xcur]
		delatY = sample[ycur+3] - sample[ycur]
		delatT = sample[tcur+3] - sample[tcur]
		xcur += 3
		ycur += 3
		tcur += 3
		if delatT == 0:
			delatT = 0.001

		avgxAcc += delatX / delatT**2
		avgyAcc += delatY / delatT**2
		avgAcc  += math.sqrt(delatX**2 +  delatY**2) / delatT**2
		count += 1
	
	avgxAcc = avgxAcc / count
	avgyAcc = avgyAcc / count
	avgAcc = avgAcc / count

	return [xAccelation, yAccelation, Accelation, avgxAcc, avgyAcc, avgAcc]


def fitTime2x(sample, sid):
	X = []
	Y = []
	TIME = []
	for i in range(0, len(sample) -1 , 3):
		x = sample[i]
		y = sample[i+1]
		time = sample[i+2]

		X.append(x)
		Y.append(y)
		TIME.append(time)

	z1 = np.polyfit(TIME, X, 3)
	p1 = np.poly1d(z1)

	yvals=p1(TIME)
	return p1.c

def fitTime2Y(sample, sid):
	X = []
	Y = []
	TIME = []
	for i in range(0, len(sample) -1 , 3):
		x = sample[i]
		y = sample[i+1]
		time = sample[i+2]

		X.append(x)
		Y.append(y)
		TIME.append(time)

	z1 = np.polyfit(TIME, Y, 3)
	p1 = np.poly1d(z1)

	yvals=p1(TIME)
	return p1.c

def dataUtil(fname):
	data = open(fname)
	trainMatrix = []
	trainLabel = []
	pcount = 0
	ncount = 0
	featureMatrix = []
	count = 0

	for sample in data.readlines():
		sample = sample.replace('\n', '')
		sample = sample.replace('\r', '')
		sample = sample.split(',')
		sample = [int(item) for item in sample]
		sample = sample[9:]
		featureMatrix.append(sample[9:])

		if sample[-1] == 1:
			pcount += 1
		else:
			ncount += 1

		count += 1

	for i in range(0 , len(featureMatrix)):
		sampleTuple = []
		try:
			if len(featureMatrix[i]) <= 1:
				continue

			if list(fitTime2x(featureMatrix[i], i)) != []:
				sampleTuple += list(fitTime2x(featureMatrix[i], i))
			else:
				sampleTuple += [0, 0, 0, 0]

			if list(fitTime2Y(featureMatrix[i], i)) != []:
				sampleTuple += list(fitTime2Y(featureMatrix[i], i))
			else:
				sampleTuple += [0, 0, 0, 0]

			sampleTuple += calcuateSpeed(featureMatrix[i], i)
			sampleTuple += calcuateAcculation(featureMatrix[i], i)
			
			

			if len(sampleTuple) == 21:
				trainMatrix.append(sampleTuple)
				trainLabel.append(featureMatrix[i][-1])

		except Exception as e:
			logging.exception(e)
			raise e

	data.close()
	return trainMatrix, trainLabel

def classify(trainMatrix, trainLabel):
	X = np.array(trainMatrix)
	y = np.array(trainLabel)
	
	cv = StratifiedKFold(y, n_folds=6)  
  	classifier = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=40)
	
	mean_tpr = 0.0  
	mean_fpr = np.linspace(0, 1, 100)  
	all_tpr = []  
  
	for i, (train, test) in enumerate(cv):  
	    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])  
	    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])  
	    mean_tpr += interp(mean_fpr, fpr, tpr)          
	    mean_tpr[0] = 0.0                               
	    roc_auc = auc(fpr, tpr)  
	    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))  
  
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
  
	mean_tpr /= len(cv)                       
	mean_tpr[-1] = 1.0                       
	mean_auc = auc(mean_fpr, mean_tpr)      
	plt.plot(mean_fpr, mean_tpr, 'k--',  
	         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)  
	  
	plt.xlim([-0.05, 1.05])  
	plt.ylim([-0.05, 1.05])  
	plt.xlabel('False Positive Rate')  
	plt.ylabel('True Positive Rate')  
	plt.title('Gray image classiy image by random forest')  
	plt.legend(loc="lower right")  
	plt.show() 

def testModel(trainData, trainLabel, testData, testLabel):
	 classifier = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=40)
	 classifier.fit(np.array(trainData), np.array(trainLabel))
	 predictions = classifier.predict(np.array(testData))
	 report = classification_report(testLabel, predictions)
	 print report

if __name__ == '__main__':
	trainData,  trainLabel = dataUtil('train.txt')
	classify(trainData, trainLabel)
	f = open('train.txt')
	sampleSize = []
	for item in f.readlines():
		item = item.replace('\n', '')
		item = item.split(',')
		item = item[9:-1]
		sampleSize.append(len(item))

	plt.hist(sampleSize, bins=100)
	plt.xlim(0, 750)
	plt.title("Histogram with 'auto' bins")
	plt.show()
