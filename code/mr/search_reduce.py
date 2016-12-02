#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import cPickle
import random

l = int(sys.argv[1])
n = int(sys.argv[2])

def load_file():
    fullpath = '../../data/train-hrt_dict.pkl'
    try:
        return cPickle.load(open(fullpath,'rb'))
    except EOFError:
        return None

graph = load_file()

def build_subgraph(graph, e, l, n): 
    subgraph = {}
    for i in range(n):
        continue_rate = 0.6 
        t = e 
        if t<0:
            continue
        path=[]
        flag=0#标志是否是第一次跳出
        for j in range(l):
            if t not in graph:
                break
            if random.uniform(0,1.0)>continue_rate:
                if j==0:
                    flag=1
                break
            ran = random.randint(0,len(graph[t])-1) #随机选择一个关系，表示第ran个关系
            key = graph[t].keys()[ran]
            path.append(key)
            ran1 = random.randint(0,len(graph[t][key])-1)#随机选该关系的一个跳转实体
            t = graph[t][key][ran1]
        if flag==1:
            path.append(-1)
        path = tuple(path)
        subgraph.setdefault(path, []) 
        subgraph[path].append(t)  #统计各个path的尾实体词频
    return subgraph #返回子图

def main():
	for line in sys.stdin:
		entity_id = int(line.strip())
		subgraph = build_subgraph(graph, entity_id, l, n)
		for path in subgraph:
			print "%d\t%s\t%s" %(entity_id, path, subgraph[path])

if __name__ == '__main__':
	exit(main())
