import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

from hic import hic

###########################################################
###### ESS CLASS	 ##################################
# 
# compute essential distance between two matrices
#
###########################################################		

class ess:
	# constructor -------------------------------------
	
	def __init__(self,indir,norm,chromo,res,masked=[]):
		"This class computes a essential distance matrix of the dataset"
	
		self.input_path = indir
		self.norm	= norm
		self.chromo     = chromo
		self.res	= res
		self.masked     = masked
		
		self.filelist   = []
		self.nmatrix    = 0
		# get list of files #
		
		for directory in os.listdir(self.input_path):
			path = '%s/%s' %( self.input_path, directory)
			if os.path.isdir(path):
				self.nmatrix += 1
				filename = '%s/%s/%s_chr%d_%s.npy' % ( path, self.norm, self.norm, self.chromo, self.res )
				if os.path.exists(filename) and directory not in masked:
					self.filelist += [ filename ]
		
	def copy(self):
		"returns copy of the object"
		
		cess = ess(self.input_path, self.norm, self.chromo, self.res, self.masked)
		
		try:
			cess.max_nvec = self.max_nvec
		except AttributeError:
			pass
		
		try:
			cess.eig  = dict(self.eig)
			cess.eigv = dict(self.eigv)
		except AttributeError:
			pass
		
		try:
			cess.significant = dict(self.significant)
		except AttributeError:
			pass		

		return cess		

	# methods -----------------------------------------
	
	def get_spectra(self,nvec):
		"compute and store a list of the spectra of the hic matrices"
		
		self.max_nvec = nvec
		
		self.eig    = {}
		self.eigv   = {}
		self.banned = {}
		
		for f in self.filelist:
			
			tmp_hic = hic(f)
			
			print(tmp_hic.refname)

			eig,eigv = tmp_hic.get_spectrum(nvec)
			sums     = tmp_hic.matrix.sum(axis=0)
			
			self.eig[tmp_hic.refname]    = eig
			self.eigv[tmp_hic.refname]   = eigv
			self.banned[tmp_hic.refname] = (sums==0)
				
	def test_random(self,pvalue=1):
		"compute the significant eigenvectors"
		
		def cdf(x,avg,sig):
			return 0.5*(1+sp.special.erf((x-avg)/np.sqrt(2*sig)))
	
		def kolmogorov_smirnov(eigv,p_thres):

			K = eigv.shape[0]	
			N = eigv.shape[1]

			p = np.zeros(K)

			for i in range(K):
				p[i]  = sp.stats.kstest(eigv[i],cdf,(0,1./N))[1]
	
			ind   = p<p_thres*1./N
	
			return ind

		
		self.significant = {}

		for i in range(1,self.nmatrix+1):
			key = 'hic%03d' % i
			if key in self.eigv.keys():
			
				eigv = np.copy(self.eigv[key])
				
				indv = self.banned[key]
				inde = (np.abs(self.eig[key])>1E-5)
				
				eigv = eigv[:,~indv]
			
				ind = kolmogorov_smirnov(eigv,pvalue)
				self.significant[key] = ind*inde
		
		return self.significant
		
	def get_pseudo_spectra(self, nvec, npseudo=-1,from_norm='nrm'):
		"compute and store a list of spectra of hic matrices and of pseudoreplicates"
		
		self.max_nvec = nvec

		self.eig    = {}
		self.eigv   = {}
		self.banned = {}
		
		cell2ref  = {}
		ref2cell  = {}

		metafile  = '%s/metadata.txt' % self.input_path
			
		for f in self.filelist:
			
			tmp_hic = hic(f)
			
			eig,eigv = tmp_hic.get_spectrum(nvec)
			sums     = tmp_hic.matrix.sum(axis=0)
			
			self.eig[tmp_hic.refname]    = eig
			self.eigv[tmp_hic.refname]   = eigv
			self.banned[tmp_hic.refname] = (sums==0)
			
			ref2cell[tmp_hic.refname]  = tmp_hic.cell
			
			if tmp_hic.cell not in cell2ref.keys():
				cell2ref[tmp_hic.cell] = [tmp_hic.refname]
			else:
				cell2ref[tmp_hic.cell]+= [tmp_hic.refname]
				  
		cell_counts = { cell:len(cell2ref[cell]) for cell in cell2ref.keys() }
		
		cell_max    = max(cell_counts, key = lambda x: cell_counts.get(x) )
		nmax        = cell_counts[cell_max]
		
		if npseudo < 0:
			npseudo = nmax
			
		nonpseudo = self.nmatrix
		
		opt = open( 'pseudo_metadata.txt', 'w')
		
		if os.path.exists(metafile):
			ipt = open(metafile,'r')
			opt.write(ipt.read())
			ipt.close()
		
		for c in cell2ref.keys():
			
			if len(cell2ref[c]) > 1:
				original = cell2ref[c][:]
			
				while len(cell2ref[c]) < npseudo:
				
					ref = np.random.choice(original,2,replace=False)
				
					fn1 = '%s/%s/%s/%s_chr%d_%s.npy' % ( self.input_path, ref[0], from_norm, from_norm, self.chromo, self.res )
					fn2 = '%s/%s/%s/%s_chr%d_%s.npy' % ( self.input_path, ref[1], from_norm, from_norm, self.chromo, self.res )
				
					tmp_hic1 = hic(fn1)
					tmp_hic2 = hic(fn2)
				
					tmp_hic  = pseudo(tmp_hic1,tmp_hic2,np.random.uniform(0.,1.))
				
					refname  = 'hic%03d' % (self.nmatrix+1)
				
					self.nmatrix += 1
				
					eig,eigv = tmp_hic.get_spectrum(nvec)
			
					self.eig[refname]  = eig
					self.eigv[refname] = eigv	
				
					ref2cell[refname]       = tmp_hic.cell
					cell2ref[tmp_hic.cell] += [ refname ]
					
					opt.write('%s\tpseudo\t%s\n' % ( refname, ref2cell[refname] ) )
		
		opt.close()
		
	def get_essential_distance(self,output,nvec=-1,spectrum='flat'):
		"compute and save a essential distance matrix"
		
		opt = open(output,'w')
		
		if nvec==-1:
			nvec = self.max_nvec
		
		for i in range(1,self.nmatrix+1):
			key_i = 'hic%03d' % i
			if not key_i in self.eig.keys():
				opt.write('%d\t%d\t%s\n' % ( i,i,'-' ) )
			else:
				opt.write('%d\t%d\t%.10f\n' % ( i,i,0. ) )
			
				print(key_i, key_i)
			
			for j in range(i+1,self.nmatrix+1):
				key_j = 'hic%03d' % j
				if not key_j in self.eig.keys() or not key_i in self.eig.keys():
					opt.write('%d\t%d\t%s\n' % ( i,j,'-' ) )
				else:
				
					print(key_i, key_j)
				
					if nvec == 'test':
						
						eigv1 = self.eigv[key_i][self.significant[key_i]]
						eigv2 = self.eigv[key_j][self.significant[key_j]]
						
						eig1  = self.eig[key_i][self.significant[key_i]]
						eig2  = self.eig[key_j][self.significant[key_j]]
					
					else:
						eigv1 = self.eigv[key_i][:nvec]
						eigv2 = self.eigv[key_j][:nvec]
					
						eig1  = self.eig[key_i][:nvec]
						eig2  = self.eig[key_j][:nvec]
					
					overlap = np.dot(eigv1,eigv2.T)**2
					
					if spectrum == 'flat':					
						
						eig1  = np.sign(eig1)/np.sqrt(len(eig1))
						eig2  = np.sign(eig2)/np.sqrt(len(eig2))
										
						E2,E1 = np.meshgrid(eig2,eig1)
					
						dist  = np.sqrt(max([1. - np.sum(E1*E2*overlap),0.]))
											
					elif spectrum == 'norm':
						
						eig1  = eig1/np.sqrt(np.dot(eig1,eig1))
						eig2  = eig2/np.sqrt(np.dot(eig2,eig2))
						
						E2,E1 = np.meshgrid(eig2,eig1)
						
						dist  = np.sqrt(max([1. - np.sum(E1*E2*overlap),0.]))
					
					elif spectrum == 'none':
						
						E2,E1 = np.meshgrid(eig2,eig1)
						
						dist = np.sqrt(max([1.-np.sum(E1*E2*overlap),0.]))
						
					else:
						try:
							k     = float(spectrum)
						
							eig1  = eig1/np.sum(np.abs(eig1)**(1./k))**k
							eig2  = eig2/np.sum(np.abs(eig2)**(1./k))**k
							
							E2,E1 = np.meshgrid(eig2,eig1)
						
							dist  = np.sqrt(max([np.sum(eig1**2+eig2**2) - 2.*np.sum(E1*E2*overlap), 0.]))	
							
						except ValueError:
							
							eig1  = np.sign(eig1)/np.sqrt(len(eig1))
							eig2  = np.sign(eig2)/np.sqrt(len(eig2))
							
							E2,E1 = np.meshgrid(eig2,eig1)
							
							dist  = np.sqrt(max([1. - np.sum(E1*E2*overlap),0.]))
						
					opt.write('%d\t%d\t%.10f\n' % ( i,j,dist) )
					
