from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time

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

def sk_SVM(trainX, trainY, testX, testY):
	'''使用支持向量机进行分类，返回准确率'''
	clf = svm.SVC(gamma=0.0001, C=5)
	clf.fit(trainX, trainY)
	return clf.score(testX, testY)

def choseBestParam(trainX, trainY):
	print("Searching the best parameters for SVC ...")
	param_grid = {'C': [1, 5, 10, 50], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
	clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=2, n_jobs=4)# 参数n_jobs=4表示启动4个进程
	clf = clf.fit(trainX, trainY)
	print("Best parameters found by grid search:")
	print(clf.best_params_)

if __name__ == '__main__':
	trainX, trainY, testX, testY = get_dataSet()
	trainX = normalization_data(trainX)
	testX = normalization_data(testX)
	for k in range(200, 300, 10):
		start_time = time.time()
		p_trainX, p_testX = pca_data(trainSet=trainX, testSet=testX, k=k)
		Accuracy = sk_SVM(trainX=p_trainX, trainY=trainY, testX=p_testX, testY=testY)
		print('n_components: {}, Accuracy: {:.3f}%, Time: {:.2f}s...'.format(k, Accuracy*100, time.time()-start_time))




