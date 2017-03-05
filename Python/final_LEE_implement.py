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
    nbrs.fit(X.T)
    dis, ind = nbrs.kneighbors(X.T)
    return ind

def find_all_weights(X, n_neighbours=5):
    """ Find local interpolation weights for all the points"""
    sample_num = X.shape[1]
    weights_matrix = np.zeros((sample_num, sample_num))
    nbrs = nearest_neighbours(X, n_neighbours + 1) # +1 NEEDED bucause deleting 1 afterwards
    # Delete the first coulumn since the closest to a point is itself
    nbrs = np.delete(nbrs, 0, axis=1)
    # Cycle through all the ponits
    for i_index in range(sample_num):
        cur_x_i = X[:, i_index]
        cur_near_points = X[:, nbrs[i_index, :]]
        cur_weights = find_local_weights(X,i_index, cur_near_points)
        weights_matrix = fill_weight_matrix(weights_matrix, cur_weights, nbrs[i_index,:], i_index)
    return weights_matrix

def find_local_weights(X, i_index, near_points):
    """ """
    C_matrix = create_C_matrix(X, i_index, near_points)
    b = np.ones(near_points.shape[1])
    print "b shape"
    print b.shape
    print "C shape"
    print C_matrix.shape
    w = linalg.solve(C_matrix, b)
    # NORMALIZE
    w_sum = np.sum(w)
    w_norm = np.divide(w, w_sum)
    # check
    return w

def regularization_matrix(C, small_denom=10000000):
    trace = np.trace(C)
    Delta = trace / small_denom
    reg = np.multiply(Delta, np.identity(C.shape[0]))
    return reg

def create_C_matrix(X, i_index, near_points):
    """ Create the C matrix for point x_i """
    # subtract to each COLUMN xi
    x_i = X[:, i_index]
    print 'near points'
    print near_points.shape
    Z_T = np.subtract(near_points, x_i)
    Z = Z_T.T
    # Covariance matrix
    C = np.dot(Z, Z.T)
    # Add regularization
    reg = regularization_matrix(C)
    regularized_C = C + reg
    return regularized_C

def fill_weight_matrix(weights_matrix, w, nbrs_index, index):
    w = np.array(w)
    nbrs_index = np.array(nbrs_index)
    weights_matrix[index, nbrs_index] = w
    return weights_matrix

# Step 2: M matrix -------------------------------------------------------------
def create_M(weights_matrix):
    identity = np.identity(weights_matrix.shape[0])
    diff = identity - weights_matrix
    M = np.dot(diff.T, diff)
    return M

# Step 3: Find the eigenvectors ------------------------------------------------
# i.e. get the 'u' vectors
def get_u(M, n_components):
    eigenvalues, eigenvectors = linalg.eig(M) # PROBLEM: IS EIG THE CORRECT FUNCTION??
    # Sort Eigencalues/eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # drop the eigenvector associated to the smallest eigenvalue
    correct_eigenvectors = np.delete(eigenvectors, 0, axis=0)
    # Select the first n_components eigenvectors
    u = eigenvectors[1:n_components+1, :]
    return u

# Step 4: Multiply by the weights to obtain correct y representations ----------
def get_transformed_points(W, u):
    # u = np.dot(W,u) >>>>>>>>>>>?<<<<<<<<<<<
    return u

# Step 5: Wrap Up --------------------------------------------------------------
def fit_LLE(X, n_neighbours, n_components):
    X = X.T # Transpose to fit the correct algorithm
    W = find_all_weights(X, n_neighbours)
    M = create_M(W)
    print M.shape
    # Get the matrix containing as ROWS the vectors embedded in the n_components space
    u = get_u(M, n_components)
    # y = get_transformed_points(W, u) >>>>>>>>>>>>???<<<<<<<<<<<<<
    y = u.T
    return y
