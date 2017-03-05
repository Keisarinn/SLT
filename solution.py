import numpy as np
import sys
from mnist import MNIST
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt


def load_mnist(N):
	mndata = MNIST('./mnistdata')
	mndata.load_training()
	mndata.load_testing()
	X = np.array(mndata.train_images)
	y = np.array(mndata.train_labels)
	return X[1:N],y[1:N]

def lle(X,y,K,d):
	#1: compute nearest neighboors of each datapoint
	knn = NearestNeighbors(n_neighbors=K+1,metric='chebyshev')
	knn.fit(X)
	nn = knn.kneighbors(X,return_distance=False)
	W = np.zeros([len(X),len(X)])
	for i in range(0,len(X)):
		Ci = np.zeros([K,K])
		for j in range(1,K+1):
			for k in range(1,K+1):
				Ci[j-1,k-1] = np.dot((X[i]-X[nn[i][j]]),(X[i]-X[nn[i][k]]))
		Ci_inv = np.linalg.inv(Ci)
		den = np.sum(Ci_inv)
		for j in range(0,K):
			W[i,nn[i][j+1]] = np.sum(Ci_inv[j,:])/den
	M = np.dot((np.identity(len(X))-W).transpose(),(np.identity(len(X))-W))
	eig_vals,Y = eigh(M,eigvals=(1,d))
	return Y,M

def plot2D(Y,labels):
	fig = plt.figure()
	plt.scatter(Y[:,0],Y[:,1],c=labels,marker='o',s=50)
	plt.colorbar()
	return plt

def plot3D(Y,labels):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	p = ax.scatter(Y[:,0],Y[:,1],Y[:,2],c=labels,marker='o',s=50)
	plt.colorbar(p)
	return plt

def main():
	d = int(sys.argv[1])
	N = int(sys.argv[2])
	X,y = load_mnist(N)
	for K in range(0,10):
		Y,M = lle(X,y,K,d)
		if d==2:
			plt = plot2D(Y,y)
		else:
			plt = plot3D(Y,y)
		plt.title('LLE using '+str(K)+' nearest neighbors. Chebyshev distance.')
		plt.savefig('fig_'+str(d)+'d_'+str(K)+'k_cheby.png')
		print(K)

if __name__=='__main__':
	main()



#U, s, V = np.linalg.svd(M, full_matrices=True)

