import numpy as np 
from sklearn.neighbors import NearestNeighbors 

def fit_transform(X, n_neighbors=5, n_components=2, metric='euclidean', small_denominator=1000):
	# X := N by D data matrix
	# W := N by N weight matrix
	# Y := N by D data matrix
	N = X.shape[0]
	W = np.zeros((N, N))

	# using Nearest Neighbor Algorithm
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='kd_tree').fit(X)
	dist, indx = nbrs.kneighbors(X)

	# compute W matrix
	for i in range(N):
		v = X[i,:]
		# could use kd-tree to find it faster
		# ind = np.argpartition(np.linalg.norm(X-v, 2, 1), n_neighbors+1)[1:n_neighbors+1]
		ind = indx[i,:]
		temp = X[ind,:] - v
		C = temp.dot(temp.T)
		C_reg = C + ((np.trace(C) / small_denominator) * np.eye(C.shape[0]))
		C_inv = np.linalg.inv(C_reg)
		W[i, ind] = C_inv.sum(1) / C_inv.sum()

	# compute M 
	I = np.eye(N)
	diff = I - W
	M = diff.T.dot(diff)

	# compute bottom d+1 eigenvalues
	eigenvalues, eigenvectors = np.linalg.eigh(M)
	Y = eigenvectors[:,1:(n_components+1)]
	return M, Y