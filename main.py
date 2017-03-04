import numpy as np
import sys
from IPython import embed

from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import solve, eigh
from scipy.sparse import csr_matrix, eye

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # # # # # # # # # # #
DATA_DOWNLOAD_DIR = './Data/'
NUM_SAMPLES = 7000

N_COMPONENTS = 2
N_NEIGHBORS = 4
DIST_FUNC = ('cosine', 'sqeuclidean')[1]
# # # # # # # # # # # #


def bary_weights(X, nbhood_X):
    n_samples, n_neighbors = nbhood_X.shape[:2]
    W = np.empty((n_samples, n_neighbors))

    for i in range(n_samples):
        D = nbhood_X[i] - X[i]
        C = D.dot(D.T)

        w = solve(C, np.ones(n_neighbors), sym_pos=True)
        W[i,:] = w / w.sum()
    return W

def barycenter_graph(X, k):
    N = X.shape[0]
    idxRanges = np.arange(0, N*k +1, k)

    knn = NearestNeighbors(k+1, n_jobs=-1).fit(X)
    #knn = NearestNeighbors(k+1, algorithm='brute', metric=DIST_FUNC).fit(X)

    nbIdx = knn.kneighbors(X, return_distance=False)[:,1:]
    data = bary_weights(X, X[nbIdx])

    return csr_matrix((data.ravel(), nbIdx.ravel(), idxRanges), shape=(N,N))

def LLE(X, n_neighbors, n_components):
    W = barycenter_graph(X, n_neighbors)
    M = eye(W.shape[0]) - W - W.T + W.T*W

    eVals, eVects = eigh(M.toarray(), eigvals=(1,n_components), overwrite_a=True)
    return -eVects/(eVals**0.5), eVals.sum()

def reverse_interp(new_y, Y, k):
    knn = NearestNeighbors(k+1, n_jobs=-1).fit(Y)
    nbDist, nbIdx = knn.kneighbors(new_y)
    nbDist, nbIdx = nbDist.ravel(), nbIdx.ravel()

    w = 1/nbDist;  w /= w.sum()
    nb_X = X[nbIdx,:]

    weighted_xs = [ w_i * X_i for w_i, X_i in zip(w, nb_X) ]
    return np.sum(weighted_xs, axis=0)

'''''''''''''''
'''''''''''''''

def get_mnist_sample(data_dir, n):
    mnist = fetch_mldata('MNIST original', data_home=data_dir)
    X, y = mnist.data.astype('float'), mnist.target

    np.random.seed(11)
    randind = np.random.choice(X.shape[0], n, replace=False)

    return X[randind,:], y[randind]

def plot_array(arr):
    x_ax = range(len(arr))
    plt.plot(x_ax, arr)
    plt.show()

def plot_components(comps, labels):
    if comps.shape[1] == 3:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(comps[:,0], comps[:,1], comps[:,2], c=labels)
    else:
        plt.scatter(comps[:,0], comps[:,1], c=labels)
    plt.show()

'''''''''''''''
'''''''''''''''


# DATA
X, y = get_mnist_sample(DATA_DOWNLOAD_DIR, NUM_SAMPLES)

# LLE
components, error = LLE(X, N_NEIGHBORS, N_COMPONENTS)

'''
errors = [LLE(X, nk, N_COMPONENTS)[1] for nk in range(1,11)]
embed();  sys.exit(0)
'''

# PLOT
plot_components(components, y)
print(error)

# PRY
embed()
