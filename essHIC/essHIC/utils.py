def mat2index(inputfile,outputfile):
	"loads the input matrix and prints it out in the index format"
	
	ipt = open(inputfile,'r')
	opt = open(outputfile,'w')
	i = 0
	
	for line in ipt:
		j = 0
		for value in line.split():
			if j >= i:
				if float(value) != 0:
					opt.write('%d\t%d\t%s\n' % (i,j,value))
			j += 1
		i += 1
	
	ipt.close()
	opt.close()
