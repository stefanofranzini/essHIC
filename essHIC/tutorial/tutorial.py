#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import os

import essHIC

###################################################################
# create experiment library with Observed over Expected matrices  #
###################################################################

mymaker = essHIC.make_hic('raw_data','hic_data') 					# initialize make_hic class with input and output paths
mymaker.get_metadata('raw_data/metadata.txt')						# list experiments and metadata

mymaker.save_data('nrm',res='100kb',dirtree='05_sub-matrices/nrm')			# saves matrices in the new directory

mymaker.save_decay_norm('nrm',res='100kb',dirtree='05_sub-matrices/nrm',makenew=True)	# saves OoE matrices in the new directory

###################################################################
# use hic class to analyze and plot an hic matrix                 #
###################################################################

hicA = essHIC.hic('hic_data/hic001/dkn/dkn_chr17_100kb.npy')	# read experiment hic001
hicB = essHIC.hic('hic_data/hic012/dkn/dkn_chr17_100kb.npy')	# read experiment hic012

print 'hicA contains experiment %s, chromosome %d, its cell-type is %s' % ( hicA.oldrefname, hicA.chromo, hicA.cell ) # print some information on hicA
print 'hicB contains experiment %s, chromosome %d, its cell-type is %s' % ( hicB.oldrefname, hicB.chromo, hicB.cell ) # print some information on hicB

hicA.get_spectrum()	# computes eigenvectors and eigenvalues for hicA
hicB.get_spectrum()	# computes eigenvectors and eigenvalues for hicB

plt.plot(np.abs(hicA.eig),'C0o',label=hicA.oldrefname)
plt.plot(np.abs(hicB.eig),'C1o',label=hicB.oldrefname)
plt.legend(fontsize=20)
plt.xlabel('eigen-number',fontsize=30)
plt.ylabel('eigenvalue',fontsize=30)
plt.tight_layout()
plt.show()		# plot of hicA and hicB spectra

hicA.plot()		# plot of hicA matrix
hicB.plot()		# plot of hicB matrix

plt.show()

hicA.reduce(10,norm='none')		# obtain the essential matrix of hicA, using 10 eigenspaces
hicB.reduce(10,norm='none')		# obtain the essential matrix of hicB, using 10 eigenspaces

hicA.plot()		# plot of hicA essential matrix
hicB.plot()		# plot of hicB essential matrix

plt.show()

###################################################################
# compute a distance matrix of all experiments                    #
###################################################################

if not os.path.exists('./dist'):	# check if the directory "dist" already exists 
	os.mkdir('./dist')		# if it does not, create it
	
	
mydistance = essHIC.ess('hic_data','dkn',17,'100kb')	# initialize the ess class to compute the essential distances 
mydistance.get_spectra(100)				# compute and store eigenspaces of all matrices in the dataset, up to the 100th

for k in range(1,100):
	mydistance.get_essential_distance('dist/dist_chr17_100kb_%d.dat' % (k),nvec=k,spectrum=1.)	# compute distance using k eigenspaces and save it to file

###################################################################
# analyze a distance matrix			                  #
###################################################################

mydist = essHIC.dist('dist/dist_chr17_100kb_10.dat', metafile='hic_data/metadata.txt')	# load the essential distance with 10 eigenspaces

mydist.get_gauss_sim()	# compute the similarity matrix

mydist.plot()	# plot the distance matrix
plt.show()

mydist.get_roc()	 	# compute ROC curve from this matrix
AUC = mydist.get_roc_area()	# compute the area under the ROC curve

print 'The area under the curve is %f' % AUC	# print the value of area under the curve
mydist.plot_roc()				# plot the ROC curve
plt.show()

mydist.hier_clustering(nclust=2)			# compute clusters (2 clusters) using hierarchical clustering (Ward method)
mydist.show_clusters()					# plot the computed clusters

mydist.spec_clustering(nclust=2)			# compute clusters (2 clusters) using spectral clustering (Ward method)
mydist.show_clusters()					# plot the computed clusters

mydist.plot_dendrogram(leafname='old')	# plot a dendrogram from the distance, use the original names of the experiments for the leaves of the tree
plt.show()

mydist.MDS()		# compute MultiDimensional Scaling for dimensional reduction
mydist.plot_MDS()	# show results in 3D
plt.show()

###################################################################
# compare results as the numbero of eigenspaces increases         #
###################################################################

rocAUC = []	# empty list of roc AUC values

for k in range(1,100):
	mydist = essHIC.dist('dist/dist_chr17_100kb_%d.dat' % (k), metafile='hic_data/metadata.txt') # load the essential matrix
	
	mydist.get_roc()			# compute roc curve for k eigenspaces distance matrix
	rocAUC += [ mydist.get_roc_area() ]	# compute roc AUC for k eigenspaces distance matrix and add to the list
	
plt.plot(range(1,100),rocAUC,'C3-',lw=2.)	# plot the list of AUC as k increases
plt.xlabel('eigen-numer',fontsize=30)
plt.ylabel('AUC', fontsize=30)
plt.tight_layout()
plt.show()

mydist2	  = essHIC.dist('dist/dist_chr17_100kb_2.dat',  metafile='hic_data/metadata.txt') # load the essential matrix with 2  eigenspaces
mydist10  = essHIC.dist('dist/dist_chr17_100kb_10.dat', metafile='hic_data/metadata.txt') # load the essential matrix with 10 eigenspaces
mydist99  = essHIC.dist('dist/dist_chr17_100kb_99.dat', metafile='hic_data/metadata.txt') # load the essential matrix with 99 eigenspaces
 
mydist2.plot_dendrogram(leafname='old') # plot a dendrogram from the distance with 2 eigenspaces
plt.show()

mydist10.plot_dendrogram(leafname='old') # plot a dendrogram from the distance with 10 eigenspaces
plt.show()

mydist99.plot_dendrogram(leafname='old') # plot a dendrogram from the distance with 99 eigenspaces
plt.show()

##################################################################

print "Congratulations! You completed the tutorial!"
