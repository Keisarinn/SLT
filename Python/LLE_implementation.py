from __future__ import division
import numpy as np
import scipy.spatial as spatial # This is needed to find the nearest neighbours of every point
import scipy.linalg as linalg

# Step 1: find weights of local patches
def find_weights(X, n_neighbours, norm=2):
    print "Asked #neighbours is: " + str(n_neighbours) + "; #datapoints is: " + str(X.shape[0])
    if n_neighbours > X.shape[0]:   # Asked for more neighbours than points
        n_neighbours = X.shape[0]-1   # Max NN is all the other points in the graph minus yourself
        print "WARNING: #neighbours bigger than #datapoints, #neighbours changed to: " + str(n_neighbours)
    weights_matrix = np.zeros((X.shape[0], X.shape[0]))
    # Create the CKD tree
    new_nn = n_neighbours +1    # This is needed to later drop the neighbours ad distance 0 (the point itself)
    ckd_tree = spatial.cKDTree(data=X, leafsize=100)
    # Cycle through all the points in X
    for index in range(X.shape[0]) :
        # Find NN
        dist, NN = ckd_tree.query(X[index], k=new_nn, p=norm) # Here Minkowsky norm can be changed
        # need to remove the closest one because it's the point itself
        correct_NN = np.delete(NN, 0)
        cur_C = create_C_matrix(index, X, correct_NN, n_neighbours)
        regul = regularization_matrix(cur_C)
        regularized_C = cur_C + regul
        # invert matrix
        cur_C_inv = np.linalg.inv(regularized_C)
        # calculate the weights for the current index
        for index_j in range(n_neighbours):
            index_in_X = correct_NN[index_j]
            weights_matrix[index][index_in_X] = calc_Wij(cur_C_inv, index_j)
    return weights_matrix

def create_Cjk(x, nj, nk):
    a = x - nj
    b = x - nk
    res = np.dot(a,b.T)
    print res.shape
    return res

def create_C_matrix(index, X, correct_NN, n_neighbours):
    C = np.zeros((n_neighbours, n_neighbours))
    nearest_neighbours = X[correct_NN,]
    for j in range(n_neighbours):
        for k in range(n_neighbours):
            C[j][k] = create_Cjk(X[index,:], nearest_neighbours[j], nearest_neighbours[k] )
    return C

def sum_numerator(C_inverse, j):
    total = 0
    for index_k in range(C_inverse.shape[0]):
        total = total + C_inverse[j][index_k]
    return total

def sum_denominator(C_inverse):
    total = 0
    for index_l in range(C_inverse.shape[0]):
        for index_k in range(C_inverse.shape[0]):
            total = total + C_inverse[index_l][index_k]
    return total

def calc_Wij(C_inverse, j):
    Wij = sum_numerator(C_inverse, j) / sum_denominator(C_inverse)
    return Wij

def regularization_matrix(C, small_denom=1000000):
    trace = np.trace(C)
    Delta = trace / small_denom
    reg = np.multiply(Delta, np.identity(C.shape[0]))
    return reg

# Step 2: compute the M matrix
def get_M(W):
    dim = W.shape[0]
    Id_M = np.identity(n=dim) - W
    M = np.dot(Id_M, Id_M)
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
    correct_eigenvectors = np.delete(eigenvectors, 0, axis=0)
    # Select the first n_components eigenvectors
    u = correct_eigenvectors[0:n_components, :]
    return u

# Step 4: Multiply by the weights to obtain correct y representations
def get_transformed_points(W, u):
    # return np.dot(W,u) >>>>>>>>>>>?<<<<<<<<<<<
    return u

# Step 5: Wrap Up
def fit_LLE(X, n_neighbours, n_components, norm=2):
    W = find_weights(X, n_neighbours)
    M = get_M(W)
    # Get the matrix containing as ROWS the vectors embedded in the n_components space
    u = get_u(M, n_components)
    y = get_transformed_points(W, u)
    return y.T
