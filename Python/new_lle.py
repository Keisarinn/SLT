from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse import eye
from scipy.linalg import eigh
from scipy.linalg import solve
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def fit_LLE(X, n_neighbors, n_components):
    W = weight_matrix(X, n_neighbors)
    M = eye(W.shape[0]) - W - W.T + W.T*W
    eigenvals, eigenvects = eigh(M.toarray(), eigvals=(1,n_components), overwrite_a=True)
    u = -eigenvects/(eigenvals**0.5)
    u = np.asarray(u)
    return u

def find_weights(X, nbrs):
    n_samples, n_neighbors = nbrs.shape[:2]
    W = np.empty((n_samples, n_neighbors))
    for i in range(n_samples):
        Z = nbrs[i] - X[i]
        # Covariance matrix
        C = Z.dot(Z.T)
        w = solve(C, np.ones(n_neighbors), sym_pos=True)
        W[i,:] = w / w.sum()
    return W

def weight_matrix(X, k):
    N = X.shape[0]
    idx = np.arange(0, N*k +1, k)
    knn = NearestNeighbors(k+1, n_jobs=-1).fit(X)
    nn_index = knn.kneighbors(X, return_distance=False)[:,1:]
    cur_weights = find_weights(X, X[nn_index])
    return csr_matrix((cur_weights.ravel(), nn_index.ravel(), idx), shape=(N,N))

def interpolation(new_y, Y, X, k):
    knn = NearestNeighbors(k+1, n_jobs=-1).fit(Y)
    neigh_dist, nn_index = knn.kneighbors(new_y)
    neigh_dist, nn_index = neigh_dist.ravel(), nn_index.ravel()
    w = 1/neigh_dist
    w /= w.sum()
    nb_X = X[nn_index,:]
    weighted_xs = [ w_i * X_i for w_i, X_i in zip(w, nb_X) ]
    return np.sum(weighted_xs, axis=0)

def interp_original_space(start, end, steps):
    step_size = np.subtract(end, start)
    step_size = np.divide(step_size, steps)
    for i in range(steps):
        new_im = np.multiply(step_size, i) + start
        plt.imshow(new_im, cmap='Spectral')
        plt.imsave("interp_orig/im_"+str(i) + ".png", new_im, format='png')
