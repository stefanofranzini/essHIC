import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

from scipy import linalg as la
from scipy import ndimage as nmg
from matplotlib import colors as mplc
import matplotlib.patheffects as PathEffects

###########################################################	
###### HIC CLASS ##########################################
# contains an hic matrix and metadata
#
###########################################################

class hic:
	# constructor -------------------------------------
	
	def __init__(self,datafile,from_pairs=True):
		"This class contains an hic contact map and its metadata"

		# obtain info from datafile

		paths = datafile.split('/')
		
		if len(paths) > 1:
		
			indir_list       = paths[:-3]
			self.indir       = ''
			for string in indir_list:
				self.indir += string + '/'
			self.refname     = paths[-3]
			
		else:
			self.indir	 = ''
			self.refname     = ''
		
		info  = paths[-1][:-4].split('_')
		
		if len(info) > 1:
			self.norm	 = info[0]
			if 'full' not in info[1]:
				self.chromo = int(info[1][3:])
			else:
				self.chromo = info[1]
			self.res	 = info[2]
			if len(info) > 3:
				self.properties = []
				for i in range(3,len(info)):
					self.properties += [ info[i] ]
		else:
			self.norm	 = 'dkn'
			self.chromo	 = 0
			self.res	 = '1kb'
		
		# load metadata
		
		filename = '%s/metadata.txt' % self.indir
		found = False
		if os.path.exists(filename):
			ipt = open(filename,'r')
			
			for line in ipt:
				fld = line.split()
				if fld[0] == self.refname:
					self.oldrefname = fld[1]
					self.cell = fld[2]
					found = True
			ipt.close()
			if not found:
				self.oldrefname = 'unknown'
				self.cell = 'unknown'
		else:
			self.oldrefname = 'unknown'
			self.cell = 'unknown'
		
		filename = '%s/chromosomes.txt' % self.indir
		found = False
		
		if os.path.exists(filename):
			ipt = open(filename,'r')
			for line in ipt:
				if self.chromo == 'full':
					break
				fld = line.split()
				if fld[0] == self.res:
					self.length = int(fld[self.chromo])
					found = True
			ipt.close()
		
		# load matrix
		
		if from_pairs:
		
			loaded = np.load(datafile)
			
			if not found:	
				self.length = int(np.max(loaded[:2]))+1

			self.matrix = np.zeros((self.length,self.length))
			
			self.matrix[loaded[0].astype(int),loaded[1].astype(int)] = loaded[2]
			self.matrix[loaded[1].astype(int),loaded[0].astype(int)] = loaded[2]
			
		else:
			self.matrix = np.load(datafile)
		
	# methods -----------------------------------------

	def resize(self,res_factor):
		"coarse grains matrix at lower resolution"
		
		if res_factor<1:
			return None
		
		new_length = self.length/res_factor + np.sign(self.length%res_factor)
		
		new_matrix = np.zeros((new_length,new_length))
		
		for i in range(new_length):
			for j in range(i,new_length):
				new_matrix[i,j] = np.mean( self.matrix[i*res_factor:(i+1)*res_factor,j*res_factor:(j+1)*res_factor] )
				new_matrix[j,i] = new_matrix[i,j]
		
		self.matrix = new_matrix
		self.length = new_length
		
		resref  = { 'kb': 1000, 'Mb': 1000000 }
		
		new_res = resref[self.res[-2:]]*int(self.res[:-2])*res_factor
		
		if new_res%1000000==0:
			self.res = '%dMb' % ( new_res/1000000 )
		else:
			self.res = '%dkb' % ( new_res/1000 )
			
	def decay_norm(self):
		"compute decay norm matrix"
		
		u = np.arange(self.length)
		
		i,j = np.meshgrid(u,u)
		
		delta = np.abs(i-j)
		
		for d in range(self.length):
			ind = delta == d
			norm = np.trace(self.matrix,offset=-d)/(self.length-d)
			if norm == 0:
				norm = 1.
			
			self.matrix[ind] /= norm
	
	def clean(self):
		"remove empty rows and columns from the matrix"
		
		sums = np.sum(self.matrix,axis=0)
		ind  = sums!=0
		
		self.matrix = self.matrix[:,ind]
		self.matrix = self.matrix[ind,:]
		
		self.length  = len(self.matrix)
	
	def downsample(self,depth_fraction):
		"downsample matrix"
		
		if self.norm not in ['nrm','raw']:		
			filename = '%s%s/nrm/nrm_chr%d_%s.npy' % ( self.indir, self.refname, self.chromo, self.res )
			 
			loaded = np.load(filename)
			
			self.matrix = np.zeros((self.length,self.length))
			
			self.matrix[loaded[0].astype(int),loaded[1].astype(int)] = loaded[2]
			self.matrix[loaded[1].astype(int),loaded[0].astype(int)] = loaded[2]		
		
		original_depth = int(np.sum(self.matrix))
		new_depth      = int(np.sum(self.matrix)*depth_fraction)
		
		p = np.triu(self.matrix/original_depth)
		
		self.matrix    = np.random.binomial(new_depth,p).astype(float)
		
		self.matrix   += np.triu(self.matrix,k=1).T
		
		self.decay_norm()
		
	def vc_norm(self):
		"compute vc norm matrix"
		
		sums = np.sum(self.matrix,axis=0)
		sums[sums==0] = 1
		
		x,y  = np.meshgrid(sums,sums)
	
		self.matrix = self.matrix/(x*y)
	
	def vcsqrt_norm(self):
		"compute vc sqrt norm matrix"
		
		sums = np.sqrt(np.sum(self.matrix,axis=0))
		sums[sums==0] = 1

		x,y  = np.meshgrid(sums,sums)

		self.matrix = self.matrix/(x*y)
		
	def pearson(self):
		"compute pearson coeff matrix"
		
		self.matrix  = np.corrcoef(self.matrix)
		self.matrix[np.isnan(self.matrix)] = 0.
	
	def smooth(self,sigma=2.):
		"compute gaussian smoothing of the matrix"
		
		self.matrix = nmg.gaussian_filter(self.matrix,sigma=sigma)
		
	def laplacian(self):
		"computes the laplacian of the matrix"

		rowsum = self.matrix.sum(axis=1)

		D  = np.diag(rowsum)
		D_ = np.zeros(D.shape)
		D_[D>0] = 1./np.sqrt(D[D>0])

		L = D - self.matrix
		
		self.matrix = np.dot(D_,np.dot(L,D_))
		
	def reduce(self,nvec=10,order='abs',norm=1.):
		"computes the essential matrix"
		
		sums = self.matrix.sum(axis=0)
		null = (sums==0)
		
		self.get_spectrum(nvec=nvec,order=order)
		self.norm_spectrum(norm=norm)
		self.matrix = np.dot(self.eigv[:nvec].T,np.dot(np.diag(self.eig[:nvec]),self.eigv[:nvec]))

		self.matrix[null,:] = 0.
		self.matrix[:,null] = 0.
		
	def get_spectrum(self,nvec=-1,order='abs'):
		"computes the nvec highest eigenvalues (in absolute value)"

		if 2*nvec > self.length:
			nvec = -1

		if nvec == -1:
			eig,eigv = np.linalg.eigh(self.matrix)
			eigv = eigv.T
		else:
			hi = self.length-1
			lo = self.length-nvec
			eig,eigv = la.eigh(self.matrix,eigvals=(lo,hi))
			eigv = eigv.T
			
			hi = nvec-1
			lo = 0
			eig_,eigv_ = la.eigh(self.matrix,eigvals=(lo,hi))
			eigv_ = eigv_.T
			
			eigv = np.concatenate((eigv_,eigv))
			eig  = np.concatenate((eig_,eig))
		
		if order == 'abs':
			ind = np.argsort(np.abs(eig))[::-1]
		elif order == 'sgn':
			ind = np.argsort(eig)[::-1]
		eig = eig[ind]
		eigv = eigv[ind]
		
		if nvec != -1:
			self.eig  = eig[:nvec]
			self.eigv = eigv[:nvec]
		else:
			self.eig  = eig
			self.eigv = eigv
		
		return self.eig,self.eigv
		
	def norm_spectrum(self,norm=1.0):
		"normalizes spectrum"
		
		if norm == 'flat':
			
			self.eig = np.sign(self.eig)/np.sqrt(len(self.eig))
		
		elif norm == 'norm':
			
			self.eig = self.eig/np.sqrt(np.dot(self.eig,self.eig))
		
		elif norm == 'none':
			
			self.eig = self.eig
		
		else:
			try:
				k = float(norm)
				
				self.eig = self.eig/np.sum(np.abs(self.eig)**(1./k))**k
			
			except ValueError:
				self.eig = np.sign(self.eig)/np.sqrt(len(self.eig))
	
	def print_matrix(self,save):
		"prints the matrix as list of indexes"
		
		u = np.arange(self.length)
		i,j = np.meshgrid(u,u)
		ind = self.matrix!=0
				
		triu= np.triu_indices(self.length)

		ind = ind[triu]
		i   = i[triu]
		j   = j[triu]
		mat = self.matrix[triu]
				
		N   = len(ind[ind])
		toprint = np.zeros( (3,N) )
		
		toprint[0] = i[ind]
		toprint[1] = j[ind]
		toprint[2] = mat[ind]
		
		np.save(save, toprint )	
	
	# plotters --------------------------------------------	
	
	def plot(self,vmax=2.5, vmin=0.0, cmap='Reds', plotkind='flat', cbar=False, triangle=False,ax='none'):
		"plot the matrix"
		
		setax = False
		
		if ax=='none':
			setax  = True
			fig,ax = plt.subplots(1,1,figsize=(8,8))
		
		if triangle:
			matrix = np.tril(self.matrix)
			matrix[np.triu_indices(len(matrix),k=1)] = np.nan
			matrix = np.ma.masked_invalid(matrix)			
			
			if plotkind=='flat':
		
				cs = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
		
			elif plotkind=='log':
				cs = ax.imshow(matrix,norm=mplc.LogNorm(),cmap=cmap)
			elif plotkind=='bilog':
				cs = ax.imshow(matrix,norm=mplc.SymLogNorm(linthresh=0.03, linscale=0.03),cmap=cmap)
			else:
				return 1
			
			for axis in ['top','right']:
				ax.spines[axis].set_linewidth(0.)
		else:
			if plotkind=='flat':
		
				cs = ax.imshow(self.matrix, vmin=vmin, vmax=vmax, cmap=cmap)
		
			elif plotkind=='log':
				z = np.ma.masked_where(self.matrix <= 0, self.matrix)
			
				cs = ax.imshow(z,norm=mplc.LogNorm(),cmap=cmap)
			elif plotkind=='bilog':
				z = np.ma.masked_where(self.matrix==0,self.matrix)
				
				cs = ax.imshow(z,norm=mplc.SymLogNorm(linthresh=0.03, linscale=0.03),cmap=cmap)
			else:
				return 1
		
		if cbar:
			fig  = plt.gcf()
			cbar = fig.colorbar(cs,extend='both')
			cbar.ax.minorticks_off()
			cbar.ax.set_position([0.86,0.25,0.1,0.65])
		
		if setax:
			ax.set_position([0.2,0.25,0.65,0.65])
			ax.set_xticks([],[])
			ax.set_yticks([],[])


	
	def plot_chromosome(self,centromere='none',regions='none',bins='none',orientation='horizontal',ticks="none"):
		'plots a chromosome cartoon alongside the figure'
		
		if centromere=='auto':
			hucen = [0.50566343, 0.38426689, 0.45636911, 0.26346053, 0.26769912, 0.35693388, 0.37720403, 0.31168831, 0.34900285, 0.29689808,
       				0.39925651, 0.27059713, 0.15687993, 0.16556914, 0.1894317 , 0.41216216, 0.30495553, 0.2260184 , 0.4153605 , 0.44070513,
				0.28144989, 0.2969697 , 0.39122014, 0.21663778]
                        
			centromere = self.length*hucen[self.chromo-1] #ciao
		
		fig = plt.gcf()
		if orientation=='horizontal':
			ax = fig.add_axes([0.19,-0.305,0.67,1.])
		elif orientation=='vertical':
			ax = fig.add_axes([-0.355,0.24,1.,0.67])
		else:
			return 1

		x = np.linspace(0.,self.length,100000)
		
		w = 0.07*self.length
		
		n = 8
		if centromere != 'none':
			y = w/self.length*(self.length**n - (2.*x-self.length)**n)**(1./n) - 0.5*w*np.exp(-np.abs(x-centromere)*70./self.length)
		else:
			y = w/self.length*(self.length**n - (2.*x-self.length)**n)**(1./n)
		
		x = np.concatenate((x,x[::-1]))
		y = np.concatenate((y,-y[::-1]))
		
		if orientation=='vertical':
			(x,y) = (y,x)
		
		plt.fill(x,y,fc='#e6d9bc',lw=0.)
		
		if regions!='none':
			for reg in regions:
				if orientation=='horizontal':
					regind  = (x>reg['bounds'][0])*(x<reg['bounds'][1])
				elif orientation=='vertical':
					regind  = (y>reg['bounds'][0])*(y<reg['bounds'][1])
						
				xreg    = x[regind]
				yreg    = y[regind]
				
				plt.fill(xreg,yreg,fc=reg['color'],lw=0.)
		if bins!='none':
			for bn in bins:
				for i in bn['bins']:
					if orientation=='horizontal':
						binind = (x>i)*(x<i+1.1)
					elif orientation=='vertical':
						binind = (y>i)*(y<i+1.1)
					
					xbin   = x[binind]
					ybin   = y[binind]
					
					plt.fill(xbin,ybin,fc=bn['color'],lw=0.)
		
		plt.fill(x,y,fc="none", lw=4.,ec='k')	

		if ticks!='none':
			myticks = np.arange(ticks,self.length,ticks)+0.5
			w_ = 0.01*self.length
			if orientation=='horizontal':
				for tic in myticks:
					plt.plot([tic]*2,[-w_,w_],'k-',lw=3)
					txt = plt.text(tic-3.5*w_,2*w_,str(int(tic-0.5)), rotation=90, weight='bold')
					txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
				txt = plt.text(self.length-3.5*w_,2*w_,str(int(self.length)), rotation=90, weight='bold')
				txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])	
			
			elif orientation=='vertical':
				for tic in myticks:
					plt.plot([-w_,w_],[tic]*2,'k-',lw=3)
					txt = plt.text(-2*w_,tic-w_,str(int(tic-0.5)), weight='bold')	
					txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
				txt = plt.text(-2*w_,self.length-w_,str(int(self.length)), weight='bold')	
				txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
				
		k = 0.01
		
		if orientation=='horizontal':
			plt.xlim([-self.length*k,self.length*(1.+k)])
		elif orientation=='vertical':
			plt.ylim([self.length*(1.+k),-self.length*k])
		
		ax = plt.gca()
		ax.set_aspect('equal')
		ax.set_axis_off()
		

class pseudo:
	# constructor -------------------------------------
	
	def __init__(self,hic1,hic2,alpha=0.5):
		"This class contains a pseudoreplicate obtained by sampling two hic maps"
		
		N = hic1.length
		
		ind = np.random.uniform(0.,1.,size=(N,N)) < alpha
		
		matrix      = np.copy(hic2.matrix)
		matrix[ind] = hic1.matrix[ind]
		
		################################# decay norm
		
		u = np.arange(N)
		
		i,j = np.meshgrid(u,u)
		
		delta = np.abs(i-j)
		
		self.matrix = np.copy(matrix)
		
		for d in range(N):
			ind = delta == d
			norm = np.trace(matrix,offset=-d)/(N-d)
			if norm == 0:
				norm = 1.
			
			self.matrix[ind] /= norm
		
		self.length = hic1.length
		self.cell   = hic1.cell
		self.res    = hic1.res
		
		self.parents= [ hic1.refname, hic2.refname ]
	
	def get_spectrum(self,nvec=-1,order='abs'):
		"computes the nvec highest eigenvalues (in absolute value)"

		if 2*nvec > self.length:
			nvec = -1

		if nvec == -1:
			eig,eigv = np.linalg.eigh(self.matrix)
			eigv = eigv.T
		else:
			hi = self.length-1
			lo = self.length-nvec
			eig,eigv = la.eigh(self.matrix,eigvals=(lo,hi))
			eigv = eigv.T
			
			hi = nvec-1
			lo = 0
			eig_,eigv_ = la.eigh(self.matrix,eigvals=(lo,hi))
			eigv_ = eigv_.T
			
			eigv = np.concatenate((eigv_,eigv))
			eig  = np.concatenate((eig_,eig))
		
		if order == 'abs':
			ind = np.argsort(np.abs(eig))[::-1]
		elif order == 'sgn':
			ind = np.argsort(eig)[::-1]
		eig = eig[ind]
		eigv = eigv[ind]
		
		if nvec != -1:
			self.eig  = eig[:nvec]
			self.eigv = eigv[:nvec]
		else:
			self.eig  = eig
			self.eigv = eigv
		
		return self.eig,self.eigv
		
	def norm_spectrum(self,norm=1.0):
		"normalizes spectrum"
		
		if norm == 'flat':
			
			self.eig = np.sign(self.eig)/np.sqrt(len(self.eig))
		
		elif norm == 'norm':
			
			self.eig = self.eig/np.sqrt(np.dog(self.eig,self.eig))
		
		elif norm == 'none':
			
			self.eig = self.eig
		
		else:
			try:
				k = float(norm)
				
				self.eig = self.eig/np.sum(np.abs(self.eig)**(1./k))**k
			
			except ValueError:
				self.eig = np.sign(self.eig)/np.sqrt(len(self.eig))

	def reduce(self,nvec=10,order='abs',norm=1.):
		"computes the essential matrix"
		
		self.get_spectrum(nvec=nvec,order=order)
		self.norm_spectrum(norm=norm)
		
		self.matrix = np.dot(self.eigv.T,np.dot(np.diag(self.eig),self.eigv))
		
	# plotters --------------------------------------------	
	
	def plot(self,vmax=2.5, vmin=0.0, cmap='Reds', plotkind='flat', cbar=False, triangle=False):
		"plot the matrix"
		
		if triangle:
			matrix = np.tril(self.matrix)
			matrix[np.triu_indices(len(matrix),k=1)] = np.nan
			matrix = np.ma.masked_invalid(matrix)			
			
			if plotkind=='flat':
		
				fig, ax = plt.subplots(1,1,figsize=(8,8))
				cs = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
		
			elif plotkind=='log':
				fig, ax = plt.subplots(1,1,figsize=(8,8))
				cs = ax.imshow(matrix,norm=mplc.LogNorm(),cmap=cmap)
			elif plotkind=='bilog':
				fig,ax = plt.subplots(1,1,figsize=(8,8))
				cs = ax.imshow(matrix,norm=mplc.SymLogNorm(linthresh=0.03, linscale=0.03),cmap=cmap)
			else:
				return 1
			
			for axis in ['top','right']:
				ax.spines[axis].set_linewidth(0.)
		else:
			if plotkind=='flat':
		
				fig, ax = plt.subplots(1,1,figsize=(8,8))
				cs = ax.imshow(self.matrix, vmin=vmin, vmax=vmax, cmap=cmap)
		
			elif plotkind=='log':
				z = np.ma.masked_where(self.matrix <= 0, self.matrix)
			
				fig, ax = plt.subplots(1,1,figsize=(8,8))
				cs = ax.imshow(z,norm=mplc.LogNorm(),cmap=cmap)
			elif plotkind=='bilog':
				z = np.ma.masked_where(self.matrix==0,self.matrix)
				
				fig,ax = plt.subplots(1,1,figsize=(8,8))
				cs = ax.imshow(z,norm=mplc.SymLogNorm(linthresh=0.03, linscale=0.03),cmap=cmap)
			else:
				return 1
		
		if cbar:
			cbar = fig.colorbar(cs,extend='both')
			cbar.ax.minorticks_off()
			cbar.ax.set_position([0.86,0.25,0.1,0.65])
		
		ax.set_position([0.2,0.25,0.65,0.65])
		ax.set_xticks([],[])
		ax.set_yticks([],[])


	
	def plot_chromosome(self,centromere='none',regions='none',bins='none',orientation='horizontal',ticks="none"):
		'plots a chromosome cartoon alongside the figure'
		
		if centromere=='auto':
			hucen = [0.50566343, 0.38426689, 0.45636911, 0.26346053, 0.26769912, 0.35693388, 0.37720403, 0.31168831, 0.34900285, 0.29689808,
       				 0.39925651, 0.27059713, 0.15687993, 0.16556914, 0.1894317 , 0.41216216, 0.30495553, 0.2260184 , 0.4153605 , 0.44070513,
                                 0.28144989, 0.2969697 , 0.39122014, 0.21663778]
                        
			centromere = self.length*hucen[self.chromo-1]
		
		fig = plt.gcf()
		if orientation=='horizontal':
			ax = fig.add_axes([0.19,-0.305,0.67,1.])
		elif orientation=='vertical':
			ax = fig.add_axes([-0.355,0.24,1.,0.67])
		else:
			return 1

		x = np.linspace(0.,self.length,100000)
		
		w = 0.07*self.length
		
		n = 16
		if centromere != 'none':
			y = w/self.length*(self.length**n - (2.*x-self.length)**n)**(1./n) - 0.5*w*np.exp(-np.abs(x-centromere)*70./self.length)
		else:
			y = w/self.length*(self.length**n - (2.*x-self.length)**n)**(1./n)
		
		x = np.concatenate((x,x[::-1]))
		y = np.concatenate((y,-y[::-1]))
		
		if orientation=='vertical':
			(x,y) = (y,x)
		
		plt.fill(x,y,fc='#e6d9bc',lw=0.)
		
		if regions!='none':
			for reg in regions:
				if orientation=='horizontal':
					regind  = (x>reg['bounds'][0])*(x<reg['bounds'][1])
				elif orientation=='vertical':
					regind  = (y>reg['bounds'][0])*(y<reg['bounds'][1])
						
				xreg    = x[regind]
				yreg    = y[regind]
				
				plt.fill(xreg,yreg,fc=reg['color'],lw=0.)
		if bins!='none':
			for bn in bins:
				for i in bn['bins']:
					if orientation=='horizontal':
						binind = (x>i)*(x<i+1.1)
					elif orientation=='vertical':
						binind = (y>i)*(y<i+1.1)
					
					xbin   = x[binind]
					ybin   = y[binind]
					
					plt.fill(xbin,ybin,fc=bn['color'],lw=0.)
		
		plt.fill(x,y,fc="none", lw=4.,ec='k')	

		if ticks!='none':
			myticks = np.arange(ticks,self.length,ticks)+0.5
			w_ = 0.01*self.length
			if orientation=='horizontal':
				for tic in myticks:
					plt.plot([tic]*2,[-w_,w_],'k-',lw=3)
					txt = plt.text(tic-3.5*w_,2*w_,str(int(tic-0.5)), rotation=90, weight='bold')
					txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
				txt = plt.text(self.length-3.5*w_,2*w_,str(int(self.length)), rotation=90, weight='bold')
				txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])	
			
			elif orientation=='vertical':
				for tic in myticks:
					plt.plot([-w_,w_],[tic]*2,'k-',lw=3)
					txt = plt.text(-2*w_,tic-w_,str(int(tic-0.5)), weight='bold')	
					txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
				txt = plt.text(-2*w_,self.length-w_,str(int(self.length)), weight='bold')	
				txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
				
		k = 0.01
		
		if orientation=='horizontal':
			plt.xlim([-self.length*k,self.length*(1.+k)])
		elif orientation=='vertical':
			plt.ylim([self.length*(1.+k),-self.length*k])
		
		ax = plt.gca()
		ax.set_aspect('equal')
		ax.set_axis_off()		
