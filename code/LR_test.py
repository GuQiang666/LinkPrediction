#-*- coding:utf-8 -*-
import cPickle
import os
import sys
import cPickle
from sklearn.externals import joblib
import time
import scipy
import numpy as np
from numpy import *
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def main():
	test_data = cPickle.load(open('../data/test_data.pkl','r'))
	Y_test = cPickle.load(open('../data/test_label.pkl','r'))
	model_folder = "./model/"
	model_short_path = model_folder + "_model.m"

	#load models
	if (os.path.exists(model_short_path)):
		model = joblib.load(model_short_path)
	else:
		print "model does not exist." % (model_short_path)
		exit(0)
	
	print 'load models succeed!'
	print 'test begin, #test= %d' %(len(Y_test))
	pred=[0]*len(Y_test)
	print 'test begin %s %d-%d' %(str(test_data.shape), sum(Y_test), len(Y_test)-sum(Y_test))
	ts_test_begin = time.time()
	Y_prob = model.predict_proba(test_data)
	Y_test=np.array(Y_test)
	AUC = metrics.roc_auc_score(Y_test, Y_prob[:, 1], sample_weight = None)
	print "AUC:		   \033[32;1m%5.3f %%\033[0m" % (100 * AUC)
	#threshold = [0.16,0.2,0.24,0.28,0.3,0.35,0.4,0.43,0.46,0.49,0.52,0.55,0.57,0.59,0.61,0.63,0.65]
	'''
	threshold = [0.2,0.24,0.28,0.3,0.35,0.4,0.43,0.46,0.49]
	for thr in threshold:
		Y_pred = np.zeros(len(Y_test))
		Y_pred_pos = Y_prob[:,1]
		pos_index = [i for i in range(len(Y_pred_pos)) if Y_pred_pos[i] > thr]
		#print pos_index
		Y_pred[pos_index] = 1
		Y_correct = Y_pred == Y_test
		print sum(Y_correct[Y_test == 1]),sum(Y_correct[Y_test == 0]),len(Y_test)
		print 'threshold = %f ---------------' %(thr)
		score = float(sum(Y_correct[Y_test == 1]) + sum(Y_correct[Y_test == 0])) / len(Y_test)
		print "Accuracy:		 \033[32;1m%5.3f %%\033[0m" % (100 * score)
		sensitivity = float(sum(Y_correct[Y_test == 1])) / sum(Y_test == 1) # true positive
		print "Sensitivity:   \033[32;1m%5.3f %%\033[0m" % (100 * sensitivity)
		specificity = float(sum(Y_correct[Y_test == 0])) / sum(Y_test == 0) # true negative
		print "Specificity:   \033[32;1m%5.3f %%\033[0m" % (100 * specificity)
		precision_pos = float(sum(Y_correct[Y_pred == 1])) / sum(Y_pred == 1)
		print "Precision (pos):  \033[32;1m%5.3f %%\033[0m" % (100 * precision_pos)
		precision_neg = float(sum(Y_correct[Y_pred == 0])) / sum(Y_pred == 0)
		print "Precision (neg):  \033[32;1m%5.3f %%\033[0m" % (100 * precision_neg)

		score = model.score(test_data, Y_test, sample_weight = None)
		#AUC = metrics.roc_auc_score(Y_test, Y_prob[:, 1], sample_weight = None)
		ts_test_end = time.time()
		print 'testing successful! (%d s)' %(ts_test_end - ts_test_begin)
		#print "AUC:		   \033[32;1m%5.3f %%\033[0m" % (100 * AUC)
		print "Accuracy:		 \033[32;1m%5.3f %%\033[0m" % (100 * score)
	'''
if __name__ == '__main__':
	exit(main())
