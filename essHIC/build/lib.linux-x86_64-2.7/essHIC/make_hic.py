import numpy as np
import os

###########################################################
###### MAKE_HIC CLASS #####################################
#
# formats data and creates metadata file
#
###########################################################

class make_hic:
	# constructor--------------------------------------
	def __init__(self,indir,outdir,loader='from_index', chromosomes='human'):
		"This class allows you to create a working directory"
		
		self.input_directory = indir
		self.output_directory = outdir
		
		if loader == 'from_index':
			self.loader = self.load_from_index
		if loader == 'from_matrix':
			self.loader = self.load_from_matrix
							
		# get list of experiments
		
		name_list = os.listdir(self.input_directory)
		
		self.list_directory = []
		
		for name in name_list:
			if os.path.isdir('./%s/%s' % (indir,name) ):
				self.list_directory += [ name ]
		
		if chromosomes == 'human':
			self.chromosomes = [ 248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717,
					     133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285,
					     58617616, 64444167, 46709983, 50818468, 156040895, 57227415 ]
		else:	
			self.chromosomes = chromosomes
				
	# class methods
	
	def get_metadata(self, metadata):
		
		ipt = open(metadata,'r')
		
		name = []
		cell = []
		
		# get metadata
		
		for line in ipt:
			fld = line.split()
			if fld[0] in self.list_directory:
				name += [ fld[0] ]
				cell += [ fld[1] ]
		
		# order by cell line
		
		index = np.argsort(cell)
		
		# create dictionaries
		
		self.newref_to_oldref = {}
		self.oldref_to_newref = {}
		
		self.newref_to_cell = {}
		self.oldref_to_cell = {}
		
		k = 1
		
		for i in index:
			myname = 'hic%03d' % k
			
			k += 1
			
			self.newref_to_oldref[myname] = name[i]
			self.oldref_to_newref[name[i]]= myname
			
			self.newref_to_cell[myname] = cell[i]
			self.oldref_to_cell[name[i]]= cell[i]
		
	def get_chromosomes(self,res):
		"loads the chromosomes boundaries"
		
		# read resolution
		
		res_units = { 'Gb' : 1000000000, 'Mb' : 1000000, 'kb' : 1000 }
		
		fullres   = int(res[:-2])*res_units[res[-2:]]
		
		chromo_lens = []
		
		for c in self.chromosomes:
			chromo_lens += [ int(c/fullres)+1 ]
		
		lo_bound = 0
		chromo_bound = []
		
		for i, cl in enumerate(chromo_lens):
			hi_bound = lo_bound + cl
			chromo_bound += [ [ lo_bound, hi_bound ] ]
			lo_bound = hi_bound + 1
		
		return chromo_bound
		
	#TODO: add formatting
		
	def load_from_index(self,fname):
		"loads an hic matrix using its indices"
		
		ipt = open(fname, 'r')
		
		loaded = [[],[],[]]
		
		for line in ipt:
			if line[0] != '#':
				fld = line.split()
				loaded[0] += [ int(fld[0])   ]
				loaded[1] += [ int(fld[1])   ]
				loaded[2] += [ float(fld[2]) ]
		
		return np.array(loaded)
	
	def load_from_matrix(self,fname):
		"loads an hic matrix using its matricial form"
		
		ipt = open(fname, 'r')
		
		loaded = [[],[],[]]
		
		i = 0
		for line in ipt:
			if line[0] != '#':
				fld = line.split()
				for j in range(i,len(fld)):
					if float(fld[j]) > 0:
						loaded[0] += [ i ]
						loaded[1] += [ j ]
						loaded[2] += [ float(fld[j]) ]
				i += 1
		
		return np.array(loaded)
			
	def compute_decay_norm(self,loaded,delta_max=-1,norm=[]):
		"computes the decay norm"
		
		delta = np.abs(loaded[1]-loaded[0]).astype(int)
		if delta_max < 0:
			delta_max = int(np.max(loaded[:2])+1)
		
		# check repeated indices
		
		m = np.zeros((delta_max,delta_max))
		m[loaded[0].astype(int),loaded[1].astype(int)] = loaded[2]
		z_ = m[loaded[0].astype(int),loaded[1].astype(int)]
		
		ind = (loaded[2]-z_!=0)
		m[loaded[0,ind].astype(int),loaded[1,ind].astype(int)] = loaded[2,ind]
		loaded[2] = m[loaded[0].astype(int),loaded[1].astype(int)]
		
		#########################
		
		dkn = np.copy(loaded)
		
		if len(norm)==0:
		
			for d in set(delta):
				ind  = delta==d
				norm = np.sum(loaded[2][ind])/(delta_max-d)
				if norm != 0.:
					dkn[2,ind] /= norm
		else:
			for d in set(delta):
				ind = delta==d
				if norm[d] != 0.:
					dkn[2,ind] /= norm[d]
		return dkn	
		
	def save_metadata(self):
		"saves the metadata"
		
		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
		opt = open('%s/metadata.txt' % self.output_directory, 'w')
		
		opt.write('#name\told_name\tcell_line\n')
		
		for i in range(len(self.newref_to_cell.keys())):
			name = 'hic%03d' % (i+1)
			opt.write('%s\t%s\t%s\n' % ( name, self.newref_to_oldref[name], self.newref_to_cell[name] ) )
		opt.close()
	
	def save_chromosomes(self,res_list):
		"saves chromosomes"
		
		opt = open('%s/chromosomes.txt' % self.output_directory, 'w')
		
		for res in res_list:
			opt.write('%s' % res)
			chromo_bounds = self.get_chromosomes(res)
			for c in range(len(self.chromosomes)):
				opt.write('\t%d' % ( chromo_bounds[c][1] - chromo_bounds[c][0] + 1) )
			opt.write('\n')
		opt.close()
		
	def save_data(self,from_norm='all',res='all',dirtree='',ext='abc',nameformat='NCR',full=False):
		"save the data in binary format"
		
		iN = nameformat.index('N')
		iC = nameformat.index('C')
		iR = nameformat.index('R')
		iA = [ index for index,letter in enumerate(nameformat) if letter == 'A' ]
		
		res_list = []
		
		list_of_data = np.sort([ self.oldref_to_newref[dirn] for dirn in self.list_directory ])
				
		for dirn in list_of_data:
			dir_path = "./%s/%s/%s" % ( self.input_directory, self.newref_to_oldref[dirn], dirtree )
			if os.path.exists(dir_path):
				for filename in os.listdir(dir_path):
					extlen = len(ext)
					if filename[-extlen:] == ext:
						
						info = filename[:-extlen-1].split('_')
						
						if from_norm != 'all':
							if info[iN] != from_norm:
								continue
						
						if res != 'all':
							if info[iR] != res:
								continue
						
						if info[iC] != 'full':
							if not 'chr' in info[iC]:
								info[iC] = 'chr' + info[iC]
						
						print(dirn, self.newref_to_oldref[dirn], info[iN], info[iC], info[iR])
						
						loaded = self.loader("%s/%s" % (dir_path,filename))
						
						res_list += [ info[iR] ]
						
						if info[iC] != 'full':
							
							newfilename = '%s_%s_%s' % ( info[iN], info[iC], info[iR] )
							for i in iA:
								newfilename += '_%s' % info[i]
							newfilename += '.npy'
							
							newpath = './%s/%s/%s' % ( self.output_directory, dirn, info[iN] )						
					
							if not os.path.exists(newpath):
								os.makedirs(newpath)
							np.save('%s/%s' % ( newpath, newfilename ), loaded )
						else:
							
							if full:
								newfilename = '%s_full_%s' % ( info[iN], info[iR] )
								for i in iA:
									newfilename += '_%s' % info[i]
								newfilename += '.npy'
									
								newpath = './%s/%s/%s' % ( self.output_directory, dirn, info[iN] )
								
								if not os.path.exists(newpath):
									os.makedirs(newpath)
								np.save('%s/%s' % ( newpath, newfilename), loaded )
							
							chbounds = self.get_chromosomes(info[iR])
														
							for c in range(len(self.chromosomes)):
															
								ind = (loaded[0] >= chbounds[c][0] )*(loaded[0] <= chbounds[c][1])
								ind*= (loaded[1] >= chbounds[c][0] )*(loaded[1] <= chbounds[c][1])
								
								chromo = loaded[:,ind]
								
								chromo[0] = chromo[0] - chbounds[c][0]
								chromo[1] = chromo[1] - chbounds[c][0]
								
								newfilename = '%s_chr%d_%s' % (info[iN], c+1, info[iR])
								for i in iA:
									newfilename += '_%s' % info[i]
								newfilename += '.npy'
	
								newpath = './%s/%s/%s' %( self.output_directory, dirn, info[iN] )
								
								if not os.path.exists(newpath):
									os.makedirs(newpath)
								np.save('%s/%s' % ( newpath, newfilename ), chromo )
		
			
		self.save_metadata()
		self.save_chromosomes(set(res_list))
		
									
	def save_decay_norm(self,from_norm='nrm',res='all',makenew=False,dirtree='',ext='abc',nameformat='NCR',full=False,compute_chbounds=True):
		"get decay norm and save binary format"
		
		iN = nameformat.index('N')
		iC = nameformat.index('C')
		iR = nameformat.index('R')
		iA = [ index for index,letter in enumerate(nameformat) if letter == 'A' ]
				
		list_of_data = np.sort([ self.oldref_to_newref[dirn] for dirn in self.list_directory ])
				
		for dirn in list_of_data:
		
			newpath = './%s/%s/dkn' % ( self.output_directory, dirn )
			
			dir_path = "./%s/%s/%s" % ( self.input_directory, self.newref_to_oldref[dirn], dirtree )
			
			if os.path.exists(dir_path):
				for filename in os.listdir(dir_path):
					extlen = len(ext)
					if filename[-extlen:] == ext:
						
						info = filename[:-extlen-1].split('_')
						
						if info[iN] != from_norm:
							continue
						
						if res != 'all':
							if info[iR] != res:
								continue
						
						if info[iC] != 'full':
							if not 'chr' in info[iC]:
								info[iC] = 'chr' + info[iC]

						loaded = self.loader("%s/%s" % ( dir_path,filename) )
						
						if loaded.shape[1] == 0:
							continue
						
						print(dirn, self.newref_to_oldref[dirn], info[iN], info[iC], info[iR])
						
						if info[iC] != 'full':
							
							if compute_chbounds:
								chbounds = self.get_chromosomes(info[iR])
								try:
								
									if info[iC][3:] == 'X':
										c = len(self.chromosomes)-2
									elif info[iC][3:] == 'Y':
										c = len(self.chromosomes)-1
									else:
										c = int(info[iC][3:])-1
									
									delta_max = chbounds[c][1] - chbounds[c][0] + 1
								
								except IndexError:
									delta_max = -1
							else:
								delta_max = -1
								
							newfilename = 'dkn_%s_%s' % ( info[iC], info[iR] )
							for i in iA:
								newfilename += '_%s' % info[i]
							newfilename += '.npy'
							
							if not makenew:
								if os.path.exists('%s/%s' % ( newpath, newfilename) ):
									continue
							
							if not os.path.exists(newpath):
								os.makedirs(newpath)
							
							
							decay = self.compute_decay_norm(loaded,delta_max)

							np.save('%s/%s' % ( newpath, newfilename), decay )

						else:
							if full:
								newfilename = '%s_full_%s' % ( info[iN], info[iR] )
								for i in iA:
									newfilename += '_%s' % info[i]
								newfilename += '.npy'
								newpath = './%s/%s/dkn' % ( self.output_directory, dirn )
								
								decay = self.compute_decay_norm(loaded,-1)
								
								if not os.path.exists(newpath):
									os.makedirs(newpath)
								np.save('%s/%s' % ( newpath, newfilename), decay )
							
							if compute_chbounds:	
								chbounds = self.get_chromosomes(info[iR])
							
							for c in range(len(self.chromosomes)):
								if compute_chbounds:
									try:
										delta_max = chbounds[c][1] - chbounds[c][0] + 1
									except IndexError:
										delta_max = -1
								else:
									delta_max = -1
															
								ind = (loaded[0] >= chbounds[c][0] )*(loaded[0] <= chbounds[c][1])
								ind*= (loaded[1] >= chbounds[c][0] )*(loaded[1] <= chbounds[c][1])
								
								chromo = loaded[:,ind]
								
								if chromo.shape[1] == 0:
									continue
								
								chromo[0] = chromo[0] - chbounds[c][0]
								chromo[1] = chromo[1] - chbounds[c][0] 
								
								newfilename = 'dkn_chr%d_%s.npy' % ( c+1, info[iR])
								for i in iA:
									newfilename += '_%s' % info[i]
								newfilename += '.npy'
								
								if not makenew:
									if os.path.exists('%s/%s' % ( newpath, newfilename) ):
										continue
								
								
								if not os.path.exists(newpath):
									os.makedirs(newpath)
								
								decay = self.compute_decay_norm(chromo,delta_max)
								
								np.save('%s/%s' % ( newpath, newfilename ), decay )
								
					
	def save_common_decay_norm(self,from_norm='nrm',res='all',chromo='all',makenew=False,dirtree='',ext='abc',nameformat='NCR',full=False,compute_chbounds=True):
		"get decay norm and save binary format"
		
		iN = nameformat.index('N')
		iC = nameformat.index('C')
		iR = nameformat.index('R')
		iA = [ index for index,letter in enumerate(nameformat) if letter == 'A' ]
				
		list_of_data = np.sort([ self.oldref_to_newref[dirn] for dirn in self.list_directory ])
		
		avg_decay = 0.
		nmatrix   = 0
		
		for dirn in list_of_data:
		
			newpath = './%s/%s/dkn' % ( self.output_directory, dirn )
			
			dir_path = "./%s/%s/%s" % ( self.input_directory, self.newref_to_oldref[dirn], dirtree )
			
			if os.path.exists(dir_path):
				for filename in os.listdir(dir_path):
					extlen = len(ext)
					if filename[-extlen:] == ext:
						
						info = filename[:-extlen-1].split('_')
						
						if info[iN] != from_norm:
							continue
						
						if res != 'all':
							if info[iR] != res:
								continue
						
						if info[iC] != 'full':
							if not 'chr' in info[iC]:
								info[iC] = 'chr' + info[iC]
							
							if info[iC][3:] == str(chromo):
							
								loaded = self.loader("%s/%s" % ( dir_path,filename))
								
								if loaded.shape[1] == 0:
									continue
							
								if compute_chbounds:
									chbounds = self.get_chromosomes(info[iR])
									try:
										if info[iC][3:] == 'X':
											c = len(self.chromosomes)-2
										elif info[iC][3:] == 'Y':
											c = len(self.chromosomes)-1
										else:
											c = int(info[iC][3:])-1
								
										delta_max = chbounds[c][1] - chbounds[c][0] + 1
								
									except IndexError:
										delta_max = -1
								
								delta = np.abs(loaded[1]-loaded[0]).astype(int)
								if delta_max < 0:
									delta_max = int(np.max(loaded[:2])+1)
							
								m = np.zeros((delta_max,delta_max))
								m[loaded[0].astype(int),loaded[1].astype(int)] = loaded[2]
								z_ = m[loaded[0].astype(int),loaded[1].astype(int)]
		
								ind = (loaded[2]-z_!=0)
								m[loaded[0,ind].astype(int),loaded[1,ind].astype(int)] = loaded[2,ind]
								loaded[2] = m[loaded[0].astype(int),loaded[1].astype(int)]
							
								dkn = np.zeros(delta_max)
							
								for d in set(delta):
									ind = delta==d
									dkn[d] = np.sum(loaded[2][ind])/(delta_max-d)
							
								avg_decay += dkn
								nmatrix   += 1

						else:
							loaded = self.loader("%s/%s" % ( dir_path,filename))
								
							if loaded.shape[1] == 0:
								continue
								
							if str(chromo) == 'X':
								c = len(self.chromosomes)-2
							elif str(chromo) == 'Y':
								c = len(self.chromosomes)-1
							else:
								c = int(info[iC][3:])-1
						
							if compute_chbounds:
								try:
									delta_max = chbounds[c][1] - chbounds[c][0] + 1
							
								except IndexError:
									delta_max = -1		
							
								
							ind = (loaded[0] >= chbounds[c][0] )*(loaded[0] <= chbounds[c][1])
							ind*= (loaded[1] >= chbounds[c][0] )*(loaded[1] <= chbounds[c][1])
							
							loaded = loaded[:,ind]							
												
							if loaded.shape[1] == 0:
								continue
							
							loaded[0] = loaded[0] - chbounds[c][0]
							loaded[1] = loaded[1] - chbounds[c][0]
								
							delta = np.abs(loaded[1]-loaded[0]).astype(int)
							if delta_max < 0:
								delta_max = int(np.max(loaded[:2])+1)
							
							m = np.zeros((delta_max,delta_max))
							m[loaded[0].astype(int),loaded[1].astype(int)] = loaded[2]
							z_ = m[loaded[0].astype(int),loaded[1].astype(int)]
		
							ind = (loaded[2]-z_!=0)
							m[loaded[0,ind].astype(int),loaded[1,ind].astype(int)] = loaded[2,ind]
							loaded[2] = m[loaded[0].astype(int),loaded[1].astype(int)]
							
							dkn = np.zeros(delta_max)
							
							for d in set(delta):
								ind = delta==d
								dkn[d] = np.sum(loaded[2][ind])/(delta_max-d)
							
							avg_decay += dkn
							nmatrix   += 1
		
		avg_decay = avg_decay/nmatrix
		
		for dirn in list_of_data:
		
			newpath = './%s/%s/dkn' % ( self.output_directory, dirn )
			
			dir_path = "./%s/%s/%s" % ( self.input_directory, self.newref_to_oldref[dirn], dirtree )
			
			if os.path.exists(dir_path):
				for filename in os.listdir(dir_path):
					extlen = len(ext)
					if filename[-extlen:] == ext:
						
						info = filename[:-extlen-1].split('_')
						
						if info[iN] != from_norm:
							continue
						
						if res != 'all':
							if info[iR] != res:
								continue
						
						if info[iC] != 'full':
							if not 'chr' in info[iC]:
								info[iC] = 'chr' + info[iC]
						
						if info[iC][3:] != str(chromo):
							continue

						loaded = self.loader("%s/%s" % ( dir_path,filename) )
						
						if loaded.shape[1] == 0:
							continue
							
						
						print(dirn, self.newref_to_oldref[dirn], info[iN], info[iC], info[iR])
						
						if info[iC] != 'full':
							
							if compute_chbounds:
								chbounds = self.get_chromosomes(info[iR])
								try:
								
									if info[iC][3:] == 'X':
										c = len(self.chromosomes)-2
									elif info[iC][3:] == 'Y':
										c = len(self.chromosomes)-1
									else:
										c = int(info[iC][3:])-1
									
									delta_max = chbounds[c][1] - chbounds[c][0] + 1
								
								except IndexError:
									delta_max = -1
							else:
								delta_max = -1
								
							newfilename = 'dkn_%s_%s' % ( info[iC], info[iR] )
							for i in iA:
								newfilename += '_%s' % info[i]
							newfilename += '.npy'
							
							if not makenew:
								if os.path.exists('%s/%s' % ( newpath, newfilename) ):
									continue
							
							if not os.path.exists(newpath):
								os.makedirs(newpath)
							
							
							decay = self.compute_decay_norm(loaded,delta_max,avg_decay)

							np.save('%s/%s' % ( newpath, newfilename), decay )

						else:
							if compute_chbounds:	
								chbounds = self.get_chromosomes(info[iR])
							
							if str(chromo) == 'X':
								c = len(self.chromosomes)-2
							elif str(chromo) == 'Y':
								c = len(self.chromosomes)-1
							else:
								c = int(info[iC][3:])-1
								
							if compute_chbounds:
								try:
									delta_max = chbounds[c][1] - chbounds[c][0] + 1
								except IndexError:
									delta_max = -1
							else:
								delta_max = -1
														
							ind = (loaded[0] >= chbounds[c][0] )*(loaded[0] <= chbounds[c][1])
							ind*= (loaded[1] >= chbounds[c][0] )*(loaded[1] <= chbounds[c][1])
							
							chromo = loaded[:,ind]
							
							if chromo.shape[1] == 0:
								continue
							
							chromo[0] = chromo[0] - chbounds[c][0]
							chromo[1] = chromo[1] - chbounds[c][0] 
							
							newfilename = 'dkn_chr%d_%s.npy' % ( c+1, info[iR])
							for i in iA:
								newfilename += '_%s' % info[i]
							newfilename += '.npy'
							
							if not makenew:
								if os.path.exists('%s/%s' % ( newpath, newfilename) ):
									continue
							
							
							if not os.path.exists(newpath):
								os.makedirs(newpath)
							
							decay = self.compute_decay_norm(chromo,delta_max,avg_decay)
							
							np.save('%s/%s' % ( newpath, newfilename ), decay )																
