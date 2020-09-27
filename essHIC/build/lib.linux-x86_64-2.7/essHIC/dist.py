import numpy as np
import sklearn.cluster as sk
import sklearn.manifold as mf
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors as mplc
from mpl_toolkits.mplot3d import axes3d

from scipy.cluster.hierarchy import ward, fcluster
from scipy.cluster.hierarchy import dendrogram, linkage 
from scipy.cluster import hierarchy
from sklearn.cluster import SpectralClustering

from collections import Counter

###########################################################
###### ESS DIST		 ##################################
# 
# read and analyze hic distance matrix
#
###########################################################

class dist:
	# constructor -------------------------------------
	def __init__(self,filename,metafile=None):
		"This class stores a distance matrix and provides some analysis tools"
		
		self.metafile = metafile
		
		torm = []
		
		self.pseudo = []
		
		if self.metafile:
			
			meta = open(self.metafile,'r')
			k    = 0
			
			colors   = []
			refs     = []
			nrefs    = []
			
			for line in meta:
				if line[0] != '#':
					fld = line.split()
					
					if fld[1] == 'pseudo':
						self.pseudo += [1]
					else:
						refs += [ fld[1] ]
						nrefs+= [ fld[0] ]
						self.pseudo += [0]
					
					if len(fld) == 4:
						torm += [k]
					
					colors          += [fld[2]]
					
					k += 1

			colorset = set(colors)
			colors   = np.array(colors)
			
			counter  = 0
			
			self.col2lab = {}
			self.lab2col = {}
			
			for c in colorset:
				
				self.lab2col[c] = counter
				self.col2lab[counter] = c
				
				colors[colors==c] = counter
				counter += 1
			
			self.colors = colors.astype(int)
			self.refs   = np.array(refs)			
			self.nrefs  = np.array(nrefs)
		else:
			ipt = open(filename,'r')
		
			k = 0
		
			for line in ipt:
				if line[0] != '#':
					flds = line.split()
					k = max([k,int(flds[0])])
			ipt.close()
		
		self.pseudo = np.array(self.pseudo, dtype=bool)
		
		x = np.empty((k,k))
		x[:] = -1
		
		ipt = open(filename,'r')
		
		for line in ipt:
			if line[0] != '#':
				flds = line.split()			
				
				i = int(flds[0])-1
				j = int(flds[1])-1
				
				if flds[2] != '-' and i not in torm and j not in torm:
					x[i,j] = float(flds[2])	
					x[j,i] = x[i,j]
		
		x[np.isnan(x)] = -1
		
		self.mask = (x<0)
		self.dist = np.ma.masked_array(x,self.mask)		
		
		tomsk   = lambda M,i: np.all(M[i,np.arange(len(M))!=i]<0) 
		rowmask = np.array([ tomsk(x,i) for i in range(k) ])
		notorm  = np.arange(k)[~rowmask]
		
		self.mdist = x[notorm,:][:,notorm]
		self.mcol  = self.colors[~rowmask]
		self.mpsd  = self.pseudo[~rowmask]
		self.mrefs = self.refs[~rowmask]
		self.mnrefs= self.nrefs[~rowmask]
		
		self.mask2original = notorm
			
	# methods -------------------------------------
	
	def print_dist(self,where='prova.dat'):
		"prints the distance map"
		
		opt = open(where,'w')
		
		dummy_matrix = np.zeros(self.dist.shape)-1.
		
		for i in range(self.dist.shape[0]):
			for j in range(self.dist.shape[0]):
				dummy_matrix[i,j] = self.dist[i,j]
		
		for i in range(self.dist.shape[0]):
			for j in range(self.dist.shape[1]):
				if dummy_matrix[i,j] < 0.:
					opt.write('%d\t%d\t-\n' % ( i+1, j+1 ) )
				else:
					opt.write('%d\t%d\t%f\n' % (i+1, j+1, dummy_matrix[i,j] ) )
		
		opt.close()
		
	def order(self,ind=None):
		
		if ind is None:
			ind = np.argsort(self.colors)
			ind_= np.argsort(self.mcol)
		else:
			ind_= np.argsort(ind[self.mask2original])
		
		self.dist  = self.dist[:,ind][ind,:]
		self.mask  = self.mask[:,ind][ind,:]
		self.mdist = self.mdist[:,ind_][ind_,:]
		
		self.colors = self.colors[ind]
		self.mcol   = self.mcol[ind_]
		
		self.pseudo = self.pseudo[ind]
		self.mpsd   = self.mpsd[ind_]
		
		dni = np.arange(len(ind))[np.argsort(ind)]
		self.mask2original = np.sort(dni[self.mask2original])
		
		try:
			self.clusters = self.clusters[ind_]
		except AttributeError:
			pass

	def get_cmap(self,index=-1):
		"get similarity map from colors"
		
		cmat = 0
		
		if index==-1:
			
			ncol = self.mcol.max()+1
						
			for i in range(ncol):
				K = np.zeros(self.mcol.shape)
				K[self.mcol==i] = 1
				
				kx,ky = np.meshgrid(K,K)
				
				cmat  += kx*ky
			
		else:
			
			K = np.zeros(self.mcol.shape)
			K[self.mcol==index] = 1
			
			kx,ky = np.meshgrid(K,K)
			
			cmat += kx*ky
			
			K = np.zeros(self.mcol.shape)
			K[self.mcol!=index] = 1
			
			cmat += kx*ky
		
		return np.triu(cmat,1), np.triu(1-cmat,1)
	
	def get_roc_area(self):
		"returns area under roc curve"
		
		A = 0
	
		for i in range(1,len(self.roc[0])):
			A += (self.roc[1][i] + self.roc[1][i-1])*(self.roc[0][i] - self.roc[0][i-1])*0.5
	
		return A	  
	
	def get_roc(self,index=-1,sample=10):
		"get roc curve"
		
		C,N = self.get_cmap(index)
		
		iu = np.triu_indices(len(C),1)
		
		c = C[iu]
		n = N[iu]
		
		TPR_max = np.sum(c)
		FPR_max = np.sum(n)
		
		TPR = [0.]
		FPR = [0.]
		DST = [0.]
		
		d = self.mdist[iu]
		
		ind = np.argsort(d)
		
		c = c[ind]
		n = n[ind]
		d = d[ind]
		
		for i in range(1,len(d),sample):
		
			DST += [ d[i] ]
			TPR += [ np.sum(c[:i])/TPR_max ]		
			FPR += [ np.sum(n[:i])/FPR_max ]

		TPR += [1.]
		FPR += [1.]
		DST += [d[-1]]
		
		self.dlist = np.array(DST)
		self.roc   = np.array([ FPR, TPR ])

	def get_gauss_sim(self,gamma=None):
		"create an affinity map with a gaussian kernel"
		
		if not gamma:
			gamma = 1./(2*np.std(self.mdist))
		
		self.sim_map = np.exp(-gamma*self.mdist*self.mdist)
	
	def MDS(self,ndim=3):
		"return low dimensional representation of the points, using MDS"
		
		embedding = mf.MDS(n_components=ndim, dissimilarity='precomputed')
		self.MDSrep = embedding.fit_transform(self.mdist).T
		
		return self.MDSrep
	
	def spec_clustering(self,nclust=3):
		"compute spectral clustering"
		
		spectral = sk.SpectralClustering(n_clusters=nclust, affinity='precomputed' )
		self.clusters = spectral.fit_predict(self.sim_map)+1
		
		return self.clusters
		
	def hier_clustering(self,nclust=3):
		"compute hierarchical clustering"
		
		flat_dist = self.mdist[ np.triu_indices(len(self.mdist),1) ]
		linked = ward(flat_dist)
		self.clusters = fcluster(linked,nclust,criterion='maxclust')
		
		return self.clusters
		
	def get_dunn_score(self):
		"compute clustering dunn score"
		
		cluster_matrix = np.zeros(self.mdist.shape)

		try:
			ncls = self.clusters.max()+1
		except AttributeError:
			self.clusters = self.mcol+1
			ncls = self.clusters.max()
		
		for i in range(1,ncls+1):
			K = np.zeros(self.mcol.shape)
			K[self.clusters==i] = 1
			
			kx,ky = np.meshgrid(K,K)
			
			cluster_matrix += kx*ky
		
		cluster_matrix = np.array(cluster_matrix, dtype=bool)

		diameter   = np.max( self.mdist[cluster_matrix]  )
		separation = np.min( self.mdist[~cluster_matrix] )
		
		return separation/diameter
		
	def get_quality_score(self):
		"compute clustering quality score"
		
		try:
			ncls = self.clusters.max()
		except AttributeError:
			self.clusters = self.mcol+1
			ncls = self.clusters.max()
	
		medoids = {}
		
		for c in range(1,ncls+1):
			
			nodes_in_cluster = np.arange(len(self.mcol))[self.clusters==c]
			
			medind = np.argmin(np.sum(self.mdist[nodes_in_cluster,:][:,nodes_in_cluster],axis=1))
			
			medoids[c] = nodes_in_cluster[medind]
		
		rho = []
		
		for c in range(1,ncls+1):
			nodes_in_cluster = np.arange(len(self.mcol))[self.clusters==c]
			nodes_in_cluster = nodes_in_cluster[nodes_in_cluster!=medoids[c]]
		
			if len(nodes_in_cluster) < 1:
				continue
				
			for node in nodes_in_cluster:
				dist_to_intramedoid = self.mdist[node,medoids[c]]
			
				dist_to_intermedoid = []
				
				for c_ in range(1,ncls+1):
					if c_ != c:
						dist_to_intermedoid += [ self.mdist[node,medoids[c_]] ]
				
				dist_to_intermedoid = np.min( dist_to_intermedoid )
				
				rho += [ dist_to_intermedoid/dist_to_intramedoid ]
		
		rho = np.median(rho)
		
		return rho
		
	def get_purity_score(self):
		
		try:
			ncls = self.clusters.max()
		except AttributeError:
			self.clusters = self.mcol+1
			ncls = self.clusters.max()
		
		self.purity = 0.
		
		for nc in range(1,ncls+1):
			cluster_colors = self.mcol[self.clusters==nc]
			counter_colors = Counter(cluster_colors)

			color_max    = max(counter_colors, key = lambda x: counter_colors.get(x) )		
			self.purity += counter_colors[color_max]
		
		self.purity /= len(self.clusters)

		return self.purity
			
	# plotters -------------------------------------
	
	def plot(self,cmap='inferno',save=None):
		"plot the matrix"
		
		fig, ax = plt.subplots(1,1,figsize=(8,8))
		cax = ax.imshow(self.dist, cmap=cmap)
		
		cbar = plt.colorbar(cax)
		cbar.set_label(r"$d$",fontsize=30,rotation='horizontal')

		if save:
			plt.savefig(save, transparent=True)
		
	def plot_masked(self,cmap='inferno',save=None):
		"plot the masked matrix"
		
		fig, ax = plt.subplots(1,1,figsize=(8,8))
		cax = ax.imshow(self.mdist, cmap=cmap)
		
		#cbar = plt.colorbar(cax)
		#cbar.set_label("DISTANCE",fontsize=20,rotation='vertical')
		
		if save:
			plt.savefig(save, transparent=True)	
	
	def plot_squares(self,cmap='Spectral'):
		"plots squares boundaries for different colors"
		
		def plot_square(a,b,c='k'):
	
			x = [ a, b, b, a, a]
			y = [ a, a, b, b, a]

			plt.plot(x,y,c=c,lw=3.)
		
		regions = []
		col_reg = [self.mcol[0]]
		
		blo = 0.
		
		for i in range(len(self.mcol)):
			if self.mcol[i] != col_reg[-1]:
				regions += [ (blo-0.5,i-0.5) ]
				blo = i
				col_reg += [ self.mcol[i] ]
		
		regions += [ (blo-0.5, len(self.mcol)-0.5)]
				
		colormap = cm.get_cmap('Spectral')
		ncol     = np.max(self.mcol)+1
		col      = [ mplc.rgb2hex(colormap(i*1./(ncol-1.))) for i in range(ncol) ]
		
		for i in range(len(regions)):
			plot_square(regions[i][0],regions[i][1],c=col[col_reg[i]])
	
	def plot_similarity(self,cmap='Greys'):
		"plot knn map"
		
		fig, ax = plt.subplots(1,1,figsize=(8,8))
		ax.imshow(self.sim_map, cmap=cmap)
	
	def plot_roc(self,col='r',sim='-',lw=3.,rnd=True,save=None):
		"plot roc curve"
		
		if rnd:
			plt.plot([0,1],[0,1],'k--')
		plt.plot(self.roc[0],self.roc[1],sim, c=col, lw=lw)
		
		plt.xlim([-0.05,1.05])
		plt.ylim([-0.05,1.05])
		
		plt.xlabel(r'False Positive Rate', fontsize=25)
		plt.ylabel(r'True Positive Rate', fontsize=25)
	
		plt.tight_layout()
	
		if save:
			plt.savefig(save, transparent=True)
		
	
	def show_hist(self):
		"plot histogram of the distances"
		
		cmap = np.zeros( self.mdist.shape )
		
		for c in range(self.mcol.max()+1):
			
			k = np.zeros(self.mcol.shape)
			k[self.mcol==c] = 1
			
			kx,ky = np.meshgrid(k,k)
			
			cmap += kx*ky
					
		px,py = np.meshgrid( 1-self.mpsd.astype(int), 1-self.mpsd.astype(int) )
		omap  = px*py
		
		px,py = np.meshgrid( self.mpsd.astype(int), 1-self.mpsd.astype(int))
		pmap  = px*py
		
		ind   = np.triu_indices(self.mdist.shape[0],k=1)

		samecell_dist = np.copy(self.mdist)
		samecell_dist[ cmap*omap==0 ] = -1
		samecell_dist = samecell_dist[ind]
		samecell_dist = samecell_dist[samecell_dist>-1] 
		
		diffcell_dist = np.copy(self.mdist)
		diffcell_dist[ (1-cmap)*omap==0 ] = -1
		diffcell_dist = diffcell_dist[ind]
		diffcell_dist = diffcell_dist[diffcell_dist>-1]
		
		pseudo_dist = np.copy(self.mdist)
		pseudo_dist[ cmap*pmap==0 ] = -1
		pseudo_dist = pseudo_dist[ind]
		pseudo_dist = pseudo_dist[pseudo_dist>-1]
	
		plt.hist(samecell_dist,range=(0.,1.),bins=100,alpha=0.6, density=True)
		plt.hist(diffcell_dist,range=(0.,1.),bins=100,alpha=0.6, density=True)
		plt.hist(pseudo_dist,range=(0.,1.),  bins=100,alpha=0.6, density=True)
		plt.show()
	
	def plot_MDS(self,cmap='Spectral', save=None):
		"plot scatter plot using the MDS representation"
		
		colormap = cm.get_cmap(cmap)
		col      = [ mplc.rgb2hex(colormap(i*1./self.colors.max())) for i in range(self.colors.max()+1) ]
		
		ndim = len(self.MDSrep)
		
		fig = plt.figure(figsize=plt.figaspect(0.8)*1.5)
		
		Xoriginal = self.MDSrep[:,~self.mpsd]
		Xpseudo   = self.MDSrep[:, self.mpsd]
		
		if ndim > 3:
			n  = np.random.choice(range(3,ndim))
			ax = fig.add_subplot(111,projection='3d')
			ax.scatter(Xoriginal[0],Xoriginal[1],Xoriginal[2],marker='o', c=self.mcol[~self.mpsd], s=1000*(Xoriginal[n]-self.MDSrep[n].min()), edgecolors='k',cmap=cmap)
			ax.scatter(Xpseudo[0],Xpseudo[1],Xpseudo[2],marker='*',c=self.mcol[self.mpsd],s=1000*(Xpseudo[n]-self.MDSrep[n].min()),edgecolors='k',cmap=cmap)		
		
		if ndim == 3:
			ax = fig.add_subplot(111,projection='3d')
			ax.scatter(Xoriginal[0],Xoriginal[1],Xoriginal[2],marker='o',c=self.mcol[~self.mpsd],s=100,edgecolors='k',cmap=cmap)
			ax.scatter(Xpseudo[0],Xpseudo[1],Xpseudo[2],marker='*',c=self.mcol[self.mpsd],s=100,edgecolors='k',cmap=cmap)
		if ndim == 2:
			ax = fig.add_subplot(111)
			ax.scatter(Xoriginal[0],Xoriginal[1],marker='o',c=self.mcol[~self.mpsd],s=100,edgecolors='k',cmap=cmap)
			ax.scatter(Xpseudo[0],Xpseudo[1],marker='*',c=self.mcol[self.mpsd],s=100,edgecolors='k',cmap=cmap)				
		if ndim == 1:
			ax = fig.add_subplot(111)
			ax.scatter(Xoriginal[0],Xoriginal[1],marker='o',c=self.mcol[~self.mpsd],s=100,edgecolors='k',cmap=cmap)
			ax.scatter(Xpseudo[0],marker='*',c=self.mcol[self.mpsd],s=100,edgecolors='k',cmap=cmap)

		for c in range(self.colors.max()+1):
			ax.plot([],[],marker='o',ms=10,mew=0,c=col[c],lw=0, label=self.col2lab[c])
		ax.plot([],[],marker='*',ms=10,mew='1',c='w',mec='k',label='pseudoreplicates')
		
		#plt.legend(loc=2)
		
		ax.set_axis_off()
		
		if save:
			plt.savefig(save, transparent=True)
		
	def show_clusters(self,cmap='Spectral', save=None):
		"show clusters"
		
		def clock(hour):
			"returns next hour on the clock"
	
			hour[0] += 1
	
			if hour[0] == hour[2]:
				hour[0] = 0
				hour[1] += 1
	
			if hour[1] == 6:
				hour[1] = 0
				hour[2]+= 1
	
			return hour
	
		def read(hour):
			"returns n,m from hour"
	
			base = [ ( 1, 0),
				 ( 0, 1),
				 (-1, 1),
				 (-1, 0),
				 ( 0,-1),
				 ( 1,-1),
				 ( 1, 0) ]
				 
			n = ( hour[2]-hour[0] )*base[hour[1]][0] + hour[0]*base[hour[1]+1][0]
			m = ( hour[2]-hour[0] )*base[hour[1]][1] + hour[0]*base[hour[1]+1][1]
	
			return n,m
	
		def hexagonal(N,L,offset=[0.,0.]):
			"define hexagonal lattice with N points and lattice constant L"

			lattice = [ ]

	
			a = np.array([0.,L])
			b = np.array([np.sqrt(0.75)*L, 0.5*L])
			offset = np.array(offset)
	
			lattice += [ 0.*a+0.*b + offset ]

			hour = [0,0,1]

			while len(lattice)<N :
				n,m  = read(hour)
		
				lattice += [ n*a + m*b + offset ]
		
				hour = clock(hour)

			lattice = np.transpose(lattice)
	
			return lattice
	
		colormap = cm.get_cmap(cmap)
		col      = [ mplc.rgb2hex(colormap(i*1./(self.mcol.max()))) for i in range(self.mcol.max()+1) ]
		lab      = self.lab2col.keys()
	
		cnum     = self.clusters.max()+1
		
		cpos     = hexagonal(cnum,10)

		cluster_colors = [ self.mcol[self.clusters==c] for c in range(cnum) ]
		pseudo_markers = [ self.mpsd[self.clusters==c] for c in range(cnum) ] 
		
		fig = plt.figure(figsize=plt.figaspect(1.)*1.5)
		ax  = fig.add_subplot(111)
		fig.patch.set_visible(False)
		ax.set_aspect('equal','datalim')
		ax.axis('off')
		
		for c in range(1,cnum):
			ccol = cluster_colors[c]
			cpsd = pseudo_markers[c]
			ppos = hexagonal(len(ccol),1,offset=[cpos[0][c],cpos[1][c]])
			
			for cc in range(self.mcol.max()+1):
				ind = (ccol==cc)*(~cpsd)
				jnd = (ccol==cc)*(cpsd)
				ax.plot(ppos[0][ind],ppos[1][ind], marker='o', lw=0, ms=10, mec='k', c=col[cc])
				ax.plot(ppos[0][jnd] ,ppos[1][jnd] , marker='*', lw=0, ms=10, mec='k', c=col[cc])
				
		for cc in range(self.mcol.max()+1):
			ax.plot([],[],marker='o',ms=10,mew=0,c=col[cc],lw=0,label=lab[cc])
		ax.plot([],[],marker='*',ms=10,mew='1',c='w',mec='k',label='pseudoreplicates')
		
		plt.legend(loc=2)
		
		if save:
			plt.savefig(save, transparent=True)	
		
		plt.show()
		
	def plot_dendrogram(self,cutoff=0.,method='ward', leafname='new', save=None):
		"compute and plot hierarchical clustering"

		ax = plt.gca()
		
		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(0)
			ax.spines[axis].set_zorder(0)

		flat_dist = self.mdist[ np.triu_indices(len(self.mdist),1) ]/self.mdist.max()

		linked    = linkage(flat_dist, method)
		
		label_list = np.arange(len(self.mdist))
		
		if leafname == 'new':
			labs = np.copy(self.mnrefs)
		elif leafname == 'old':
			labs = np.copy(self.mrefs)
		
		my_palette = cm.Set2(np.linspace(0,1,len(self.col2lab)))
		
		hierarchy.set_link_color_palette([mplc.rgb2hex(rgb[:3]) for rgb in my_palette])
		
		self.dendro = dendrogram(linked,
					orientation='top',
					labels=label_list,
					distance_sort='descending',
					color_threshold=cutoff,
					show_leaf_counts=True,
					above_threshold_color='black'
				)
		
		scramble = self.mcol[self.dendro['ivl']]
		labrable = labs[self.dendro['ivl']]
		
		cmap = cm.get_cmap('Spectral')
		
		clist = list(set(self.mcol))

		col = [ cmap(i*1./(max(clist))) for i in range(max(clist)+1)  ]

		leaves = np.arange(len(self.mdist))*10 + 5
		
		for i in clist:
			i_leaves = leaves[scramble==i]
			plt.plot(i_leaves, [0]*len(i_leaves), 'o', mec='none', c=col[i], ms=10.)
			
		for c in range(max(clist)+1):
			plt.plot([],[],marker='o',ms=10,mew=0,c=col[c],lw=0, label=self.col2lab[c])
		
		plt.xticks(leaves,labrable,fontsize=15)
		
		plt.legend(loc=2)
		
		ax.set_ylim(bottom=-0.2)
		
		plt.tight_layout()
		
		if save:
			plt.savefig(save, transparent=True)
		
		return self.dendro
