#-*- coding:utf-8 -*-
import time		  
import re		  
import os  
import sys
import codecs
import json
import shutil
import numpy as np
import random
import cPickle
from sklearn import feature_extraction, feature_selection
#import sklearn
#from sklearn.feature_extraction.text import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import shape
from numpy import linalg as la
from scipy.sparse import coo_matrix
from collections import Counter

entity2idx = cPickle.load(open('../data/FB15k_entity2idx.pkl', 'rb'))
relation2idx = cPickle.load(open('../data/FB15k_relation2idx.pkl', 'rb'))

#加载.pkl格式的数据，把磁盘存储的变量读到内存中
def load_file(datatyp):
	#pathtotal = '/Users/pro/Documents/workspace/DNNKR/src/' + path
	fullpath = '../data/'+str(datatyp)+'-hrt_dict.pkl'
	try:
		return cPickle.load(open(fullpath,'rb')) #一定要用二进制读方式打开文件，否则会报EOF错误
	except EOFError:
		return None

def pick_rel(relation,datatyp):
	#rel is the specified relation, string is the data type string, 'train' or 'test'
	f = open('../data/freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
	dat = f.readlines()
	f.close()
	pick = []
	for line in dat:
		temp = line[:-1].strip().split('\t')
		if relation == temp[1]:
			pick.append([temp[0],temp[2]])
	return pick


# 给实体构建子图，以字典形式返回子图，key是path，value是以空格分隔的出现的尾实体
# n是walkers, l是path length
def build_subgraph(hrt_dict, e, l, n):
	dist = {} #distribution
	for i in range(n):
		continue_rate = 0.6
		t = e
		if t<0:
			continue
		#要将t变成id 
		path=[]
		flag=0
		for j in range(l):
			#应该把<=l跳数的t都算进去，而不是只算l跳的
			if t not in hrt_dict:
				break
			if random.uniform(0,1.0)>continue_rate:
				if j==0:
					flag=1
				break
			ran = random.randint(0,len(hrt_dict[t])-1) #随机选择一个关系，表示第ran个关系
			key = hrt_dict[t].keys()[ran]#第ran个关系的id			
			#print hrt_dict[t]
			#print hrt_dict[t].keys()
			#print key
			#记录下该关系,存到path数组中
			path.append(key)
			ran1 = random.randint(0,len(hrt_dict[t][key])-1)
			t = hrt_dict[t][key][ran1]
			#continue_rate = continue_rate * 0.6
		if flag==1:
			path.append(-1)
		path = tuple(path)
		dist.setdefault(path, [])
		dist[path].append(t)  #统计各个path的尾实体词频
	return dist #返回链接关系字典

#合并相同key的list（展平）
def union_dict(*objs):
	_keys = set(sum([obj.keys() for obj in objs],[]))
	_total = {}
	for _key in _keys:
		_total[_key] = sum([obj.get(_key,[]) for obj in objs],[])
	return _total

# 构建tfidf矩阵，返回一个长度为2的列表，[0]是去重的尾实体集，用来索引。[1]是tfidf权重矩阵，特征（路径）数*尾实体数
# dic是对应实体子图的字典
def build_tfidf(dic, voca):
	path_set = dic.keys()
	corpus = []
	for i in dic.keys(): #keys是以关系id为元素的元组
		corpus.append(dic[i].strip()) #strip函数功能是去除头尾空格
	vectorizer = CountVectorizer(vocabulary = voca)
	#该类会统计每个词语的tf-idf权值
	transformer = TfidfTransformer()
	#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
	tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
	#获取词袋模型中的所有词语  
	word = vectorizer.get_feature_names()
	#将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
	weight = tfidf.toarray()
	#print vectorizer.vocabulary_
	return word,weight,path_set

#要返回所有feature（左右路径乘积）的特征值
#参数是左右子图，字典形式。实体vocabulary
def feature_selection(index, l_subgraph, r_subgraph, voca):
	feature = []
	_,l_tfidf_weight,l_path_set = build_tfidf(l_subgraph, voca)
	_,r_tfidf_weight,r_path_set = build_tfidf(r_subgraph, voca)
	#print len(l_path_set),len(r_path_set)
	for i in range(len(l_path_set)):
		for j in range(len(r_path_set)):
			cos = cosSimilar(l_tfidf_weight[i], r_tfidf_weight[j])
			if(cos != 0):
				f = []
				f.append(index)
				f.append(str(l_path_set[i]) + '-' + str(r_path_set[j]))
				f.append(cos)
				feature.append(f)
	return feature
	
def cosSimilar(inA,inB):  
	inA=np.mat(inA)  
	inB=np.mat(inB)  
	num=float(inA*inB.T)  
	denom=la.norm(inA)*la.norm(inB)  
	#return 0.5+0.5*(num/denom)
	return num/denom

def gen_negative_pairs(pairs, n):	
	pos_pair_set = set([])
	neg_pair_set = set([])
	for elem in pairs:
		pos_pair_set.add(tuple(elem))
	for i in range(n):
		x = random.randint(0,len(pairs)-1)
		y = random.randint(0,len(pairs)-1)
		neg_pair_set.add((pairs[x][0],pairs[y][1]))
	neg_pair_set = neg_pair_set - pos_pair_set
	return list(neg_pair_set)

def get_cos(a, b):
	insection = 0
	ma, mb = 0, 0
	for x in a:
		if x in b:
			insection = insection + a[x]*b[x]
	if insection == 0:
		return 0
	for i in a.values():
		ma = ma + i*i
	for i in b.values():
		mb = mb + i*i
	return insection/(ma**(1./2)*mb**(1./2))


def get_path_score_dict_single(hrt_dict, train_rel, l_path, n, samples_single, path2id_dict):
	l = len(train_rel)
	l_sub_new, r_sub_new = {},{}
	path_score_dict_list = []
	if samples_single > l:
		samples_single = l
	s = random.sample(range(l),samples_single)
	cc=0
	for i in s: #range(l-1)
		l_sub_new = build_subgraph(hrt_dict, entity2idx.get(train_rel[i][0], -1),l_path,n)#词频
		r_sub_new = build_subgraph(hrt_dict, entity2idx.get(train_rel[i][1], -1),l_path,n)
		for key in l_sub_new.keys():
			l_sub_new[key] = dict(Counter(l_sub_new[key])) #将数组转化成词频统计字典
		for key in r_sub_new.keys():
			r_sub_new[key] = dict(Counter(r_sub_new[key])) #将数组转化成词频统计字典
		#融合两条路径，得到特征值
		path_score_dict = {}
		for l_key in l_sub_new:
			for r_key in r_sub_new:
				sim = get_cos(l_sub_new[l_key], r_sub_new[r_key])
				if sim != 0:
					idx = path2id_dict.get(str(l_key) + '-'  + str(r_key), -1)
					if idx != -1:
						path_score_dict[idx] = sim
		if path_score_dict == {}:
			cc+=1
		path_score_dict_list.append(path_score_dict)

	return path_score_dict_list, cc

mat = []

def get_feature_mat(sample, j):
	row = j * np.ones((len(sample),))
	col = sample.keys()
	val = sample.values()
	return row, col, val
	#data = coo_matrix((val, (row, col)), shape=(1, len(sample)))

	
def main():	
	n_walkers = int(sys.argv[1])
	n_samples_single = int(sys.argv[2])
	#n_rel = int(sys.argv[4])

	hrt_dict = load_file('train')
	print 'load triples succeed! #triples is %d' %(len(hrt_dict))

	path2id_dict = cPickle.load(open('../data/path2id_dict.pkl','r'))
	print 'load path info succeed!'
	#构造test集的data和label
	print 'generate feature matrix begin...'
	print 'parameter: #walkers=%d, #samples=%d' %(n_walkers, n_samples_single)
	i = 0
	row, col, val = [], [], []
	test_label = []
	relation_list=['/award/award_nominee/award_nominations./award/award_nomination/award_nominee','/film/film/release_date_s./film/film_regional_release_date/film_release_region','/award/award_nominee/award_nominations./award/award_nomination/award','/award/award_category/nominees./award/award_nomination/award_nominee','/people/person/profession','/people/profession/people_with_this_profession']

	for rel in [relation_list[4]]:
		train_rel = pick_rel(rel, 'test')
		path_score_dict_list, cc = get_path_score_dict_single(hrt_dict, train_rel, 4, n_walkers, n_samples_single, path2id_dict)
		if len(path_score_dict_list) != len(train_rel):
			print "length error---%d---%d" %(len(train_rel), len(train_rel)-len(path_score_dict_list))
		for elem in path_score_dict_list:
			cur_row, cur_col, cur_val = get_feature_mat(elem, i)
			row = np.hstack((row, cur_row))
			col = np.hstack((col, cur_col))
			val = np.hstack((val, cur_val))
			test_label.append(1)
			i = i + 1

		negative_pairs = gen_negative_pairs(train_rel,i)
		path_score_dict_list_neg, dd = get_path_score_dict_single(hrt_dict, negative_pairs, 4, n_walkers, n_samples_single, path2id_dict)
		if len(path_score_dict_list_neg) != len(negative_pairs):
			print "length error---%d---%d" %(len(negative_pairs), len(negative_pairs)-len(path_score_dict_list_neg))
		for elem in path_score_dict_list_neg:
			cur_row, cur_col, cur_val = get_feature_mat(elem, i)
			row = np.hstack((row, cur_row))
			col = np.hstack((col, cur_col))
			val = np.hstack((val, cur_val))
			test_label.append(0)
			i = i + 1

	#n_col = max(col) + 1	
	n_col = len(path2id_dict)
	test_data = coo_matrix((val, (row, col)), shape=(i, n_col))
	print 'generate feature matrix succeed! size = %d * %d' %(i, n_col)
	f = open('../data/test_data.pkl','w')
	cPickle.dump(test_data, f)
	f.close()
	g = open('../data/test_label.pkl','w')
	cPickle.dump(test_label, g)
	g.close()
	print "test feature matrix save succeed!"

if __name__ == '__main__':
	exit(main())

