from __future__ import division
import numpy as np
import scipy.spatial as spatial # This is needed to find the nearest neighbours of every point

# def my_LLE(X, n_neighbours, n_components):
#     X_transfor
#     return X_transfor

# Step 1: find weights of local patches
def find_weights(X, n_neighbours):
    weights_matrix = np.zeros((X.shape[0], X.shape[0]))
    # Create the CKD tree
    new_nn = n_neighbours +1
    ckd_tree = spatial.cKDTree(data=X, leafsize=100) # IMPORTANT: CAN USE DIFFERENT NORMS HERE!!!
    # Cycle through all the points in X
    for index in range(X.shape[0]) :
        # Find NN
        dist, NN = ckd_tree.query(X[index], k=new_nn)
        # need to remove the closest one because it's the point itself
        correct_NN = np.delete(NN, 0)
        cur_C = create_C_matrix(index, X, correct_NN, n_neighbours)
        # invert matrix
        cur_C_inv = np.linalg.inv(cur_C)
        # calculate the weights for the current index
        for index_j in range(n_neighbours):
            index_in_X = correct_NN[index_j]
            weights_matrix[index][index_in_X] = calc_Wij(cur_C_inv, index_j)
    return weights_matrix

def create_Cjk(x, nj, nk):
    a = x - nj
    b = x - nk
    res = np.dot(a,b)
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

# Step 2: compute the M matrix
