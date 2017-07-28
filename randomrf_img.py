from helper import *
from feature import *

if __name__ == "__main__":
	ds = Dataset()
	traindata, trainlabels = ds.img2matrix('pic', 1)
	
	ds1 = Dataset()
	ds1.img2matrix('samples', 1)
	testdata, testlabels = ds1.next_batch(1984)

	classifier = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=40)
	classifier.fit(traindata, trainlabels)
	classifier.fit(testdata, testlabels)
	predictions = classifier.predict(np.array(testdata))
	report = classification_report(testlabels, predictions)
	print report