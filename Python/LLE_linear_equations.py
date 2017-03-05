from __future__ import division
import numpy as np
import scipy.spatial as spatial # This is needed to find the nearest neighbours of every point
import scipy.linalg as linalg
import scipy
from sklearn.neighbors import NearestNeighbors

# Step 1: find weights --------------------------------------------------------
def nearest_neighbours(X, n_neighbours=5):
    """ Find k nearset neighbours indices """
    nbrs = NearestNeighbors(n_neighbors=n_neighbours, metric='minkowski', p=2) # CHANGE NORMS HERE
    nbrs.fit(X)
    dis, ind = nbrs.kneighbors(X)
    return ind

def find_all_weights(X, n_neighbours=5):
    """ Find local interpolation weights for all the points"""
    sample_num = X.shape[0]
    weights_matrix = np.zeros((sample_num, sample_num))
    nbrs = nearest_neighbours(X, n_neighbours + 1) # +1 NEEDED bucause deleting 1 afterwards
    # Delete the first coulumn since the closest to a point is itself
    nbrs = np.delete(nbrs, 0, axis=1)
    # Cycle through all the ponits
    for i_index in range(sample_num):
        cur_x_i = X[i_index,:]
        cur_near_points = X[nbrs[i_index, :], :]
        cur_weights = find_local_weights(cur_x_i, cur_near_points)
        weights_matrix = fill_weight_matrix(weights_matrix, cur_weights, nbrs[i_index,:], i_index)
    return weights_matrix

def find_local_weights(x_i, near_points):
    """ """
    C_matrix = create_C_matrix(x_i, near_points)
    b = np.ones(near_points.shape[0])
    w = linalg.solve(C_matrix, b)
    # NORMALIZE
    w_sum = np.sum(w)
    w_norm = np.divide(w, w_sum)
    # check
    return w

def create_Cjk(x, nj, nk):
    """ Create the element (j,k) of the matrix C """
    a = x - nj
    b = x - nk
    res = np.dot(a,b.T)
    # print "res: " +str(res)
    return res

def regularization_matrix(C, small_denom=10000000):
    trace = np.trace(C)
    Delta = trace / small_denom
    reg = np.multiply(Delta, np.identity(C.shape[0]))
    return reg

def create_C_matrix(x_i, near_points):
    """ Create the C matrix for point x_i """
    nn = near_points.shape[0] # Number of nearest neighbours
    C = np.zeros((nn, nn))
    for j in range(nn):
        for k in range(nn):
            cur_jk = create_Cjk(x_i, near_points[j], near_points[k] )
            C[j][k] = cur_jk
    reg = regularization_matrix(C)
    regularized_C = C + reg
    return regularized_C

def fill_weight_matrix(weights_matrix, w, nbrs_index, index):
    w = np.array(w)
    nbrs_index = np.array(nbrs_index)
    weights_matrix[index, nbrs_index] = w
    return weights_matrix

# Step 2: M matrix
def create_M(weights_matrix):
    identity = np.identity(weights_matrix.shape[0])
    diff = identity - weights_matrix
    M = np.dot(diff.T, diff)
    return M

# Step 3: Find the eigenvectors associated with the n_components+1 smallest eigenvalues
# i.e. get the 'u' vectors
def get_u(M, n_components):
    eigenvalues, eigenvectors = linalg.eig(M) # PROBLEM: IS EIG THE CORRECT FUNCTION??
    # Sort Eigencalues/eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # drop the eigenvector associated to the smallest eigenvalue
    # correct_eigenvectors = np.delete(eigenvectors, 0, axis=0)
    # Select the first n_components eigenvectors

    # u = eigenvectors[1:n_components+1, :]
    u_ind = []
    eig = 0
    prev = 0
    i = 1
    while i < n_components + 1 :
        eig = eigenvalues[i]
        if eig != prev:
            u_ind.append(i)
            i = i+1
        prev = eig
    print u_ind
    u = eigenvectors[u_ind, :]
    print u
    return u

# Step 4: Multiply by the weights to obtain correct y representations
def get_transformed_points(W, u):
    # u = np.dot(W,u) >>>>>>>>>>>?<<<<<<<<<<<
    return u

# Step 5: Wrap Up
def fit_LLE(X, n_neighbours, n_components):
    W = find_all_weights(X, n_neighbours)
    M = create_M(W)
    # Get the matrix containing as ROWS the vectors embedded in the n_components space
    u = get_u(M, n_components)
    # y = get_transformed_points(W, u) >>>>>>>>>>>>???<<<<<<<<<<<<<
    y = np.multiply(u.T, np.sqrt(X.shape[0]))
    return y
