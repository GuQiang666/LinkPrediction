import sys
path=sys.argv[1]

f=open('trainset_%s.svm' %(path),'r')
for line in f:
	l=line[:-1].split(' ')
	if len(l)!=1:
		print l[0].strip(),
		d={}
		for i in range(1,len(l)):
			pair=l[i].strip().split(':')
			if len(pair)==2:
				d[int(pair[0])]=pair[1]
		for key in sorted(d.keys()):
			print ' %d:%s' %(key,d[key]),
		print '\n',
