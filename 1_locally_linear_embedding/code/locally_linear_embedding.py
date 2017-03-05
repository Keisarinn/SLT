"""Locally linear embedding"""

import numpy as np
from scipy.linalg import eigh, lstsq, solve
from sklearn.neighbors import NearestNeighbors


def locally_linear_embedding(X, n_components=2, n_neighbors=10,
        metric='euclidean'):
    X = X.astype('float')
    # Get number of samples and original dimension
    N, D = X.shape

    # Find the k nearest neighbors to compute the reconstruction weights
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1).fit(X)
    indx = nbrs.kneighbors(X, return_distance=False)[:,1:]

    X_neighbors = X[indx]   # N x n_neighbors x Dims

    # Initialize the array to store the weight of the nearest neighbors of each
    # point, and for each point compute the weights
    W = np.zeros((N, N), dtype=X.dtype)
    X_reconstruct = np.empty(X.shape, dtype=X.dtype)
    for i, A in enumerate(X_neighbors):
        c_x = X[i] - A  # X_i broadcasts
        C = np.dot(c_x, c_x.T)
        w_i = solve(C, np.ones(n_neighbors, dtype=X.dtype), sym_pos=True)
        w_i /= np.sum(w_i)
        W[i, indx[i]] = w_i
        X_reconstruct[i,:] = np.dot(w_i, A)

    # Find matrix M
    M = np.eye(N) - W
    M = np.dot(M.T, M)

    # Find the eigenvectors and eigenvalues of the M matrix
    eigen_values, eigen_vectors = eigh(M, eigvals=(1, n_components))
    index = np.argsort(np.abs(eigen_values))

    # Compute the embedding vectors and reconstruction error
    Y = eigen_vectors[:, index]
    embedding_error = np.sum(eigen_values)
    reconstruction_error = np.sum((X - X_reconstruct)**2)

    return Y, embedding_error, reconstruction_error, M

def embedding_reconstruction(y, Y, X, n_neighbors=10):
    """ Parametest:
    y:  Point in the embedding space to reconstruct
    Y:  Dataset points in the embedding points
    X:  Dataset points in the original space
    """
    y = y.reshape(1,-1)
    nbrs_embedding = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Y)
    indx = nbrs_embedding.kneighbors(y, return_distance=False)[0,1:]

    Y_neighbors = Y[indx,:]
    c_y = y - Y_neighbors
    C = np.dot(c_y, c_y.T)
    # W = solve(C, np.ones(n_neighbors, dtype=X.dtype), sym_pos=True)
    # W = W.reshape(1,-1)
    # W /= W.sum()
    C = np.linalg.inv(C)
    W = np.sum(C, axis=0, keepdims=True) / np.sum(C)
    # W = W / np.sum(W)

    return np.dot(W, X[indx])
