#-*- coding:utf-8 -*-
import time
import scipy
import numpy as np
from numpy import *
import cPickle
import random
import pandas as pd
from scipy.sparse import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
from sklearn import metrics

def main():
	model_folder = "./model/"
	model_short_path = model_folder + "_model.m"
	model_coef_path = model_folder + "_coef.txt"
	model_inter_path = model_folder + "_intercept.txt"
	nonzero_col_path = model_folder + "nonzero_cols"

	penalty = 'l2'
	para = 0.01
	max_iter = 100
	tol = 0.01
	solver = 'lbfgs'

	#X_trainset, y_trainset = load_svmlight_file("trainset_pra_sorted.svm")
	X_trainset, y_trainset = load_svmlight_file("trainset_sfe_sorted.svm")
	print X_trainset.shape, y_trainset.shape
	#pca=PCA(n_components=1300)
	#newX=pca.fit_transform(X_trainset.toarray())

	n = X_trainset.shape[0]
	#print newX.shape
	index=random.sample(range(n),n)
	
	rate=0.7
	#X_train = newX[index][0:int(n*rate)]
	X_train = X_trainset[index][0:int(n*rate)]
	y_train = y_trainset[index][0:int(n*rate)]
	#X_test = newX[index][int(n*rate):]
	X_test = X_trainset[index][int(n*rate):]
	y_test = y_trainset[index][int(n*rate):]

	#pca=PCA(n_components=1000)
	#newX=pca.fit_transform(X_trainset.toarray())

	#data_sub = train_data.tocsr()
	#label_sub = np.array(train_label)
	n_pos = sum(y_train)
	n_neg = y_train.shape[0] - n_pos
	print 'training begin, size=%d*%d, %d+ %d-' %(y_train.shape[0], X_train.shape[1], n_pos, n_neg)

	ts_train_begin = time.time()
	model = LogisticRegression(penalty = penalty, dual = False, tol = tol, C = para, fit_intercept = True, intercept_scaling = 1, solver = solver, max_iter = max_iter)
	model.fit(X_train, y_train)
	ts_train_end = time.time()
	print 'train successful! (%d s)' %(ts_train_end - ts_train_begin)
	
	joblib.dump(model, model_short_path)
	np.savetxt(model_coef_path, model.coef_, delimiter=',')
	np.savetxt(model_inter_path, model.intercept_, delimiter=',')	
	
	Y_prob = model.predict_proba(X_test)
	AUC = metrics.roc_auc_score(y_test, Y_prob[:, 1], sample_weight = None)
	score = model.score(X_test,y_test)
	print "Mean Accuracy:        \033[32;1m%5.3f %%\033[0m" % (100 * score)
	print "AUC:        \033[32;1m%5.3f %%\033[0m" % (100 * AUC)
if __name__ == '__main__':
	exit(main())
	
