import os
import struct
import numpy as np
import sklearn
from mnist import MNIST
from scipy.linalg import solve,eigh,lstsq

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test(X):
	print("started")
	X = np.array(X)
	clf = LocallyLinearEmbedding(5,n_components = 3)
	print("doin stuff")
	X_lle = clf.fit_transform(X)
	

def plot2D(X,Y):
	x = X[:,0]
	y = X[:,1]
	plt.scatter(x,y,c = Y,s = 20 )
	plt.colorbar()
	plt.show()

def plot3D(X,Y):
	x = X[:,0]
	y = X[:,1]
	z = X[:,2]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	p = ax.scatter(x,y,z,c = Y,s = 20)
	fig.colorbar(p)
	plt.show()
	

	

def main():
	amount = 1000
	idx_R = 10
	k = 20
	mndata = MNIST('./data')
	images,labels = mndata.load_training()
	images = np.array(images)
	#test(images)
	#print(mndata.display(images[index]))
	#Read in the data
	#print(len(images))
	w = LLE(images[:amount],3,k)
	to_reconstruct = np.random.normal(0,1,k)#0.5*w[idx_R] + 0.5* w[idx_R + 1]#w[idx_R]#np.random.normal(0,1,2)#
	R = reconstruct(to_reconstruct,w,images[:amount],k)
	draw(R,None,28,28)
	#plot2D(w,labels[:amount])	
	

def reconstruct(X,E,O,k):
	knn = NearestNeighbors(n_neighbors= k + 1, algorithm='auto',metric = 'euclidean').fit(E)
	ind = knn.kneighbors(X,return_distance=False)[:,1:].flatten()
	closest_vectors = E[ind]
	w = lstsq(closest_vectors,X)[0]
	Xorig = np.dot(w.T,O[ind])
	print(Xorig.shape)
	return Xorig

def draw(O,R,w,h):
	O = np.reshape(O,(w,h))
	f, axarr = plt.subplots(1,2)
	if R is not None:
		R = np.reshape(R,(w,h))
		axarr[0].matshow(R,cmap=plt.cm.gray)
		axarr[0].set_title('Reconstruced')	
		axarr[0].set_axis_off()
		axarr[1].matshow(O,cmap=plt.cm.gray)
		axarr[1].set_title('Original')	
		axarr[1].set_axis_off()
	else:
		plt.matshow(O,cmap=plt.cm.gray)
	plt.show()


def LLE(X,k = 30,dim = 2):
	#For each X find the nearest neighbour

	knn = NearestNeighbors(n_neighbors= k + 1, algorithm='auto',metric = 'euclidean').fit(X)
	X = knn._fit_X
	ind = knn.kneighbors(X,return_distance=False)[:,1:] #Remove first neighbour
	samples = X.shape[0]
	W = sklearn.manifold.locally_linear.barycenter_kneighbors_graph(X,k)	
	W = W.toarray()
	M = np.eye(len(X)) - W
	#print(M)
	#f, axarr = plt.subplots(2)
	#axarr[0].matshow(M,cmap=plt.cm.gray)
	M = np.dot(M.T,M)	
	v,w =  eigh(M,eigvals=(1, dim))
	print(v)
	#axarr[1].hist(v)
	#plt.show()
	idx = v.argsort()[::-1] 
	w = w[:,idx]
	v = v[idx]
	return w
		
	
		
main()
