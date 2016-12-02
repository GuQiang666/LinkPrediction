#-*- coding:utf-8 -*-
import time
import scipy
import numpy as np
from numpy import *
import cPickle
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def load_file(path):
	#pathtotal = '/Users/pro/Documents/workspace/DNNKR/src/' + path
	fullpath = '../data/' + str(path) + '.pkl'
	try:
		return cPickle.load(open(fullpath,'rb')) #一定要用二进制读方式打开文件，否则会报EOF错误
	except EOFError:
		return None

def main():
	model_folder = "./model/"
	model_short_path = model_folder + "_model.m"
	model_coef_path = model_folder + "_coef.txt"
	model_inter_path = model_folder + "_intercept.txt"
	nonzero_col_path = model_folder + "nonzero_cols"

	penalty = 'l2'
	para = 10000
	max_iter = 1000
	tol = 0.00001
	solver = 'lbfgs'

	train_data = cPickle.load(open('../data/train_data.pkl','r'))
	train_label = cPickle.load(open('../data/train_label.pkl','r'))

	#将训练集拆分成多个
	n_train = train_data.shape[0]
	#data_sub = train_data.tocsr()
	#label_sub = np.array(train_label)
	n_pos = sum(train_label)
	n_neg = n_train - n_pos
	print 'training begin, size=%d*%d, %d+ %d-' %(n_train, train_data.shape[1], n_pos, n_neg)

	ts_train_begin = time.time()
	model = LogisticRegression(penalty = penalty, dual = False, tol = tol, C = para, fit_intercept = True, intercept_scaling = 1, solver = solver, max_iter = max_iter)
	model.fit(train_data, train_label)
	ts_train_end = time.time()
	print 'train successful! (%d s)' %(ts_train_end - ts_train_begin)
	
	joblib.dump(model, model_short_path)
	np.savetxt(model_coef_path, model.coef_, delimiter=',')
	np.savetxt(model_inter_path, model.intercept_, delimiter=',')
	
if __name__ == '__main__':
	exit(main())
	
