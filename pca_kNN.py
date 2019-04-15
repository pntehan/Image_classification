from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def get_dataSet():
	'''创建数据集'''
	trainDirs = glob('./128-128梯度实验样本/No4/训练样本/*/*.bmp')
	testDirs = glob('./128-128梯度实验样本/No4/测试样本/*/*.bmp')
	trainSet = np.array([np.array(Image.open(x)).flatten() for x in trainDirs])
	testSet = np.array([np.array(Image.open(x)).flatten() for x in testDirs])
	trainLabel = np.array([x.split('\\')[1] for x in trainDirs])
	testLabel = np.array([x.split('\\')[1] for x in testDirs])
	return trainSet, trainLabel, testSet, testLabel

def normalization_data(dataSet):
	'''对数据进行标准化操作'''
	num = list(map(lambda x: (np.mean(x), np.std(x)), dataSet))
	return np.array([(dataSet[i]-num[i][0])/num[i][1] for i in range(dataSet.shape[0])])

def pca_data(trainSet, testSet, k):
	'''对数据进行降维处理'''
	pca = PCA(n_components=k)
	pca.fit(trainSet)
	return pca.transform(trainSet), pca.transform(testSet)

def sk_kNN(test_data, test_target, stored_data, stored_target):
	'''kNN算法实现分类'''
	classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
	classifier.fit(stored_data, stored_target)
	y_pred = classifier.predict(test_data)
	return classifier.score(test_data, test_target)

if __name__ == '__main__':
	trainX, trainY, testX, testY = get_dataSet()
	trainX = normalization_data(trainX)
	testX = normalization_data(testX)
	for k in range(260, 265, 1):
		p_trainX, p_testX = pca_data(trainSet=trainX, testSet=testX, k=k)
		Accuracy = sk_kNN(test_data=p_testX, test_target=testY, stored_data=p_trainX, stored_target=trainY)
		print('n_components: {}, Accuracy: {:.3f}%...'.format(k, Accuracy*100))















