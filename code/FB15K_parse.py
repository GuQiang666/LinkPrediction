#-*- coding:utf-8 -*-
import os, sys
import cPickle

import numpy as np
import scipy.sparse as sp

if 'data' not in os.listdir('../'):
    os.mkdir('../data')

def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

# 得到某个关系的训练集和测试集
# relation是要提取的关系字符串。datatyp是'train','valid'或'test'
def pick_rel(relation,datatyp):
	#rel is the specified relation, string is the data type string, 'train' or 'test'
	f = open('../data/freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
	dat = f.readlines()
	f.close()
	pick = []
	for line in dat:
		temp = line[:-1].split('\t')
		#print temp[1].split(' ')
		if relation == temp[1].split(' ')[0]:
			pick.append([temp[0].split(' ')[0],temp[2].split(' ')[0]])
	return pick


#################################################
### Creation of the entities/indices dictionnaries

np.random.seed(753)

entleftlist = []
entrightlist = []
rellist = []

for datatyp in ['train']:
#     f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    f = open('../data/freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1]) #drop the tail character '\t'
        entleftlist += [lhs[0]]
        entrightlist += [rhs[0]]
        rellist += [rel[0]]

entleftset = np.sort(list(set(entleftlist) - set(entrightlist))) #left but not right, only-left
entsharedset = np.sort(list(set(entleftlist) & set(entrightlist))) #shared
entrightset = np.sort(list(set(entrightlist) - set(entleftlist))) #right but not left
relset = np.sort(list(set(rellist)))

entity2idx = {} #dict
idx2entity = {}
relation2idx = {}
idx2relation = {}


# we keep the entities specific to one side of the triplets contiguous
idx = 0
for i in entrightset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbright = idx #num. of only right
for i in entsharedset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbshared = idx - nbright
for i in entleftset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbleft = idx - (nbshared + nbright)

print "# of only_left/shared/only_right entities: ", nbleft, '/', nbshared, '/', nbright
# add relations at the end of the dictionary
idx = 0
for i in relset:
    relation2idx[i] = idx
    idx2relation[idx] = i
    idx += 1
nbrel = idx
print "Number of relations: ", nbrel

f = open('../data/FB15k_entity2idx.pkl', 'wb')
cPickle.dump(entity2idx, f, -1)
f.close()
g = open('../data/FB15k_idx2entity.pkl', 'wb')
cPickle.dump(idx2entity, g, -1)
g.close()
ff = open('../data/FB15k_relation2idx.pkl', 'wb')
cPickle.dump(relation2idx, ff, -1)
ff.close()
gg = open('../data/FB15k_idx2relation.pkl', 'wb')
cPickle.dump(idx2relation, gg, -1)
gg.close()

'''
#################################################
### Creation of the dataset files
# creation of hrt
for datatyp in ['train', 'valid', 'test']:
    #print datatyp
#     f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    f = open('../data/freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()

    # Declare the dataset variables
    hrt_dict={}
	# Fill the sparse matrices
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        if lhs[0] in entity2idx and rhs[0] in entity2idx and rel[0] in entity2idx: 
            if hrt_dict.has_key(entity2idx[lhs[0]])==0:
                hrt_dict[entity2idx[lhs[0]]]={}
            if hrt_dict[entity2idx[lhs[0]]].has_key(entity2idx[rel[0]])==0:
			    hrt_dict[entity2idx[lhs[0]]][entity2idx[rel[0]]]=[]
            hrt_dict[entity2idx[lhs[0]]][entity2idx[rel[0]]].append(entity2idx[rhs[0]])
    #print hrt_dict[entity2idx['/m/027rn']]
    

    # Save the datasets
    if 'data' not in os.listdir('../'):
        os.mkdir('../data')
    f = open('../data/FB15k-%s-hrt_dict.pkl' % datatyp, 'wb')
    cPickle.dump(hrt_dict, f, -1)
    f.close()
'''
