import os, sys
import cPickle
import numpy as np
import scipy.sparse as sp

if 'data' not in os.listdir('../'):
	os.mkdir('../data')

entity2idx = cPickle.load(open('../data/FB15k_entity2idx.pkl', 'rb'))
relation2idx = cPickle.load(open('../data/FB15k_relation2idx.pkl', 'rb'))

def parseline(line):
	list_split = line.split()
	lhs = list_split[0].split()
	rel = list_split[1].split()
	rhs = list_split[2].split()
	if len(list_split) > 4:
		for i in range(3,len(list_split)-1):
			rhs[0] = rhs[0] + ' ' + list_split[i] 
	return lhs, rel, rhs

#################################################
### Creation of the dataset files
def prepare_hrt_dict():
	for datatyp in ['train', 'valid', 'test']:
		f = open('../data/freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
		dat = f.readlines()
		f.close()
		hrt_dict={}
		for i in dat:
			lhs, rel, rhs = parseline(i[:-1])
			lhs_id, rel_id, rhs_id = entity2idx[lhs[0]], relation2idx[rel[0]], entity2idx[rhs[0]]
			hrt_dict.setdefault(lhs_id, {})
			hrt_dict[lhs_id].setdefault(rel_id, [])
			hrt_dict[lhs_id][rel_id].append(rhs_id)
		f = open('../data/' + datatyp + '-hrt_dict.pkl', 'w')
		cPickle.dump(hrt_dict, f, -1)
		f.close()

def main():
	prepare_hrt_dict()

if __name__ == '__main__':
	exit(main())

