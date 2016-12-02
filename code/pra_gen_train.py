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
#import FB15K_parse as FB
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
idx2relation = cPickle.load(open('../data/FB15k_idx2relation.pkl', 'rb'))
global path2id_dict
path2id_dict = {}
global id2path_dict
id2path_dict = {}
global idx
idx = 0

def get_pairs(f):
	of=open(f,'r')
	pairs=[]
	for line in of:
		l=line[:-1].split('\t')
		if(len(l)==3):
			pairs.append(l)
	return pairs

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
		#print temp
		if relation == temp[1]:
			pick.append([temp[0],temp[2]])
	return pick

#hrt_dict = load_file('train')
#print hrt_dict

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
		flag=0#标志是否是第一次跳出
		for j in range(l):
			#应该把<=l跳数的t都算进去，而不是只算l跳的
			if t not in hrt_dict:
				break
			if random.uniform(0,1.0)>continue_rate:
				if j==0:
					flag=1
				break
			ran = random.randint(0,len(hrt_dict[t])-1) #随机选择一个关系，表示第ran个关系
			key = hrt_dict[t].keys()[ran]
			#print hrt_dict[t]
			#print hrt_dict[t].keys()
			#print key
			#记录下该关系,存到path数组中
			path.append(key)
			ran1 = random.randint(0,len(hrt_dict[t][key])-1)#随机选该关系的一个跳转实体
			t = hrt_dict[t][key][ran1]
			#continue_rate = continue_rate * 0.6
		if flag==1:
			path.append(-1)
		path = tuple(path)
		#if(dist.has_key(path)==False):
		#	dist[path] = ''
		dist.setdefault(path, [])
		dist[path].append(t)  #统计各个path的尾实体词频
	return dist #返回链接关系字典

'''
def union_dict(*objs):  
	_keys = set(sum([obj.keys() for obj in objs],[]))  
	_total = {}  
	for _key in _keys:  
		_total[_key] = sum([obj.get(_key,0) for obj in objs])  
	return _total
'''

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
				#print feature
	#print len(feature)
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
	neg_pair = neg_pair_set - pos_pair_set
	#print neg_pair
	return list(neg_pair)

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


def path2id(path):
	global idx
	global path2id_dict
	global id2path_dict
	if path not in path2id_dict:
		path2id_dict[path] = idx
		id2path_dict[idx] = path
		idx = idx + 1
	return path2id_dict[path]

def pra_dfs_search_path_2(hrt_dict, head, tail, path_len):
	path_list = set([])
	for a in hrt_dict[head]:
		for b in hrt_dict[head][a]:
			if b not in hrt_dict:
				continue
			elif b==tail:
				path_list.add(tuple([a]))
				continue
			for c in hrt_dict[b]:
				for d in hrt_dict[b][c]:
					if d==tail:
						path_list.add(tuple([a,c]))
						continue
	return path_list

def pra_dfs_search_path_3(hrt_dict, head, tail, path_len):
	path_list = set([])
	for a in hrt_dict[head]:
		for b in hrt_dict[head][a]:
			if b not in hrt_dict:
				continue
			elif b==tail:
				path_list.add(tuple([a]))
				continue
			for c in hrt_dict[b]:
				for d in hrt_dict[b][c]:
					if d not in hrt_dict:
						continue
					elif d==tail:
						path_list.add(tuple([a,c]))
						continue
					for e in hrt_dict[d]:
						for f in hrt_dict[d][e]:
							if f==tail:
								path_list.add(tuple([a,c,e]))
								continue
	return path_list

def pra_dfs_search_path_4(hrt_dict, head, tail, path_len):
	path_list = set([])
	for a in hrt_dict[head]:
		for b in hrt_dict[head][a]:
			if b not in hrt_dict:
				continue
			elif b==tail:
				path_list.add(tuple([a]))
				continue
			for c in hrt_dict[b]:
				for d in hrt_dict[b][c]:
					if d not in hrt_dict:
						continue
					elif d==tail:
						path_list.add(tuple([a,c]))
						continue
					for e in hrt_dict[d]:
						for f in hrt_dict[d][e]:
							if f not in hrt_dict:
								continue
							elif f==tail:
								path_list.add(tuple([a,c,e]))
								continue
							for g in hrt_dict[f]:
								for h in hrt_dict[f][g]:
									if h==tail:
										path_list.add(tuple([a,c,e,g]))
										continue
	return path_list


def pra_dfs_search_path_6(hrt_dict, head, tail, path_len):
	path=[]
	path_list = set([])
	for a in hrt_dict[head]:
		for b in hrt_dict[head][a]:
			if b not in hrt_dict:
				continue
			for c in hrt_dict[b]:
				for d in hrt_dict[b][c]:
					if d not in hrt_dict:
						continue
					for e in hrt_dict[d]:
						for f in hrt_dict[d][e]:
							if f not in hrt_dict:
								continue
							for g in hrt_dict[f]:
								for h in hrt_dict[f][g]:
									if h not in hrt_dict:
										continue
									for i in hrt_dict[h]:
										for j in hrt_dict[h][i]:
											if j not in hrt_dict:
												continue
											for k in hrt_dict[j]:
												for l in hrt_dict[j][k]:
													path.append(a)
													path.append(c)
													path.append(e)
													path.append(g)
													path.append(i)
													path.append(k)
													if l==tail and len(path)==6:
														path=tuple(path)
														path_list.add(path)
													path=[]
	return path_list

def pra_search_path(hrt_dict, h, e, l, n):
	path_list = set([])
	for i in range(n):
		t = h
		if t<0:
			continue
		#要将t变成id 
		path=[]
		for j in range(l):
			if t not in hrt_dict:
				break
			if t==e:
				break
			ran = random.randint(0,len(hrt_dict[t])-1) #随机选择一个关系，表示第ran个关系
			key = hrt_dict[t].keys()[ran]
			path.append(key)
			ran1 = random.randint(0,len(hrt_dict[t][key])-1)#随机选该关系的一个跳转实体
			t = hrt_dict[t][key][ran1]
		if t==e:
			if len(path)>0:
				path = tuple(path)
				path_list.add(path)  #统计各个path的尾实体词频
	return path_list

def get_path_score_dict_single(hrt_dict, pairs, l_path, n, r_entity_list):
	l = len(pairs)
	#print "len_train_rel= ",l
	feature_map={}
	j=1#feature index
	of1 = open('trainset_pra.svm', 'w')
	for i in range(l-1):
		h,e=entity2idx.get(pairs[i][0], -1), entity2idx.get(pairs[i][1], -1)
		#path_list = pra_dfs_search_path_4(hrt_dict, h, e, l_path)#词频
		path_list = pra_search_path(hrt_dict, h, e, l_path, n)#词频
		of1.write("%s" %(pairs[i][2]))
		for path in path_list:
			if path not in feature_map.keys():
				feature_map[path]=j
				j+=1
			#random walk
			hit=0
			hit_list=[]
			for k in range(n):
				t=h
				if t<0 or t not in hrt_dict:
					continue
				for rel in path:
					if t not in hrt_dict or rel not in hrt_dict[t]:
						break
					ran_entity=random.randint(0, len(hrt_dict[t][rel])-1)
					t=hrt_dict[t][rel][ran_entity]
				if t==e:
					hit+=1
				else:
					hit_list.append(t)
			cell=1.0*hit/n
			of1.write(" %d:%3.2f" %(feature_map[path], cell))
			#temp=dict(Counter(hit_list))
			#if e in r_entity_list:
			#	r_entity_list.remove(e)
			#
			#count=0
			#for i in list(r_entity_list):
			#	if i in temp:
			#		count+=1

			#print len(list(r_entity_list)),count

			#for i in list(r_entity_list)[0:10]:
			#	of2.write(" %d:%3.2f" %(feature_map[path], 1.0*temp.setdefault(i,0)/n))

		of1.write("\n")
	return feature_map

def get_feature_mat(sample, j):
	row = j * np.ones((len(sample),))
	col = sample.keys()
	val = sample.values()
	#print row,col,val
	return row, col, val
	#data = coo_matrix((val, (row, col)), shape=(1, len(sample)))

	
def main():
	t0=time.clock()
	n_walkers = int(sys.argv[1])
	n_path = int(sys.argv[2])

	hrt_dict = load_file('train')
	print 'load triples succeed! #triples is %d' %(len(hrt_dict))
	
	#构造训练集的data和label
	print 'generate feature matrix begin...'
	print 'parameter: #walkers=%d' %(n_walkers)
	i = 0
	row, col, val = [], [], []
	train_label = []
	#for rel in [idx2relation[3]]:
	#15998 /award/award_nominee/award_nominations./award/award_nomination/award_nominee
	#12893 /film/film/release_date_s./film/film_regional_release_date/film_release_region
	#12175 /award/award_nominee/award_nominations./award/award_nomination/award
	#12166 /award/award_category/nominees./award/award_nomination/award_nominee
	#11636 /people/person/profession
	#11547 /people/profession/people_with_this_profession
	#for rel in ['/people/person/profession']:
	pairs = get_pairs('../data/_people_person_nationality/insection_trainset_pair.txt')
	#pairs = get_pairs('../data/trainset_relation1.txt')
	#train_rel = pick_rel(rel, 'train')
	r_entity_list = set(np.array(pairs)[:,1])
	feature_map = get_path_score_dict_single(hrt_dict, pairs, n_path, n_walkers, r_entity_list)#生成路径-频次字典
	print len(feature_map)
	print "time spending %d (s)" %(int(time.clock()-t0))

if __name__ == '__main__':
	exit(main())

