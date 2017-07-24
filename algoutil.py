#coding=utf-8

import time
import random
import numpy as np
from sklearn import svm
from scipy import interp
import matplotlib as mpl 
from sklearn import tree
from sklearn.svm import SVC
from datautil import DateUtil
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import os

resultDisk = os.path.join(os.getcwd(), "result")
resPr = os.path.join(resultDisk, 'PR.png')
resTrain = os.path.join(resultDisk, 'train.png')
resTest = os.path.join(resultDisk, 'test.png')
resSample = os.path.join(resultDisk, 'Samples.png')

class algutil(object):
    """docstring for algutil"""
    def __init__(self):
        super(algutil, self).__init__()
        self.algorithm = {
            "DecisionTree": tree.DecisionTreeClassifier(),
            "SVM": SVC(kernel="linear", C=0.025),
            "AdaBoost": AdaBoostClassifier(), 
            "KNN": KNeighborsClassifier(n_neighbors=600),
            "NaiveBayes": GaussianNB(),
            "SGD": SGDClassifier(loss="hinge", penalty="l2"), 
            "RandomForest": RandomForestClassifier()
        }
    
    def classify(self, classifierName, trainData, testData):
        classifier = self.algorithm[classifierName] 
        classifier.fit(trainData["dataMartix"], trainData["labelList"])
        predictions = classifier.predict(testData["dataMartix"])
        
        report = classification_report(testData["labelList"], predictions)
        stopWords = ['precision', 'recall', 'f1-score', 'support', 'total', '/']
        report = report.split()
        report = list(report)
        for word in stopWords:
            report.remove(word)
        
        resultDict = {}
        for index in range(0, len(report), 5):
            cls        = report[index]
            precision  = float (report[index + 1])
            recall     = float (report[index + 2])
            fscore     = float (report[index + 3])
            support    = int (report[index + 4])

            resultDict[cls] = {}
            resultDict[cls]['precision'] = precision
            resultDict[cls]['recall'] = recall
            resultDict[cls]['fscore'] = fscore
            resultDict[cls]['support'] = support

        return resultDict

def plotDataSize(aData, Type):
    import matplotlib.pyplot as plt

    size = []
    for i in range(0, 10):
        size.append(list(aData['labelList']).count(i))
    xlabel = ["class" + str(i) for i in range(0, 10)]
    plt.figure()

    if Type == 'test':
        plt.title(u"Test Data Size for each class")
    else:
        plt.title(u"Train Data Size for each class")
    plt.bar(range(0, 10), size, label=xlabel)
    plt.xticks(range(0, 10), xlabel)

    if Type == 'test':
        plt.savefig(resTest, dpi=75)
    else:
        plt.savefig(resTrain, dpi=75)

    plt.close()

def plotSamples(aData):
    import matplotlib.pyplot as plt
    trainData = aData
    plt.figure()
                                                                                                                                                                        
    for i in range(1, 11):
        plt.subplot(2, 5, i)
        img = trainData[random.randint(1, 60000)]['data']
        x = np.array(img).reshape([28, 28])  
        plt.imshow(x)

    plt.savefig(resSample, dpi=75)
    plt.close()


def plotPR(ans):
    import matplotlib.pyplot as plt
    precisions = []
    recall = []
    fscore = []
    plt.figure(1)
    
    n = 10
    X = np.arange(n)+1
    for i in range(0, 10):
        precisions.append(ans[str(i)]["precision"])
        recall.append(ans[str(i)]["recall"])
        fscore.append(ans[str(i)]["fscore"])
    rects1 = plt.bar(X, precisions, width=0.4, label="precision")
    rects2 = plt.bar(X+0.4, recall, width=0.4, label="recall")
    xlabel = ["class" + str(i) for i in range(0, 10)]
    plt.xticks(X, xlabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=2)  
    plt.title('Precision and recall')
    plt.savefig(resPr, dpi=75)
    plt.close()
