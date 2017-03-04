
import numpy as np


from sklearn.neighbors import NearestNeighbors

from scipy import linalg

from LLE_plots import plot_embedding_2D, plot_embedding_3D,matrix_plot

from datetime import datetime



def LLE_implementation(X,nfeatures,nneighbors,ncomponents,metric):

    ########################################
    # Preferences
    ########################################

    if nfeatures == None:
        nfeatures = 2000

    if nneighbors == None:
        print("ERROR: NO NEIGHBORS SELECTED!")
        return

    if ncomponents == None:
        print("ERROR: NO NUMBER OF COMPONENTS SELECTED!")
        return

    ########################################
    # Computation of Nearest Neighbors
    ########################################

    print("Running LLE with " + str(nneighbors) + " neighbors...")

    neighbor_start = datetime.now()

    nbrs = NearestNeighbors(n_neighbors=nneighbors + 1,metric=metric, algorithm='kd_tree').fit(X)
    distances, nIndices = nbrs.kneighbors(X)

    correctNeighbors = nIndices[:, 1:nneighbors + 1]

    neighbor_end = datetime.now()
    delta_neighbor = neighbor_end - neighbor_start

    print("Duration of Nearest Neighbor Compuation: ", delta_neighbor)

    ########################################
    # Computation of Weights
    ########################################

    weight_start = datetime.now()

    Weights = np.zeros((nfeatures, nfeatures))

    def compute_Cjk(X, index, j, neighbors):
        Cjk = np.zeros((1, nneighbors))

        for k in range(nneighbors):
            Cjk[0, k] = (X[index] - X[neighbors[j]]).transpose().dot(X[index] - X[neighbors[k]])

        return Cjk

    def compute_C(X, index, neighbors):
        C = np.zeros((nneighbors, nneighbors))

        for j in range(nneighbors):
            C[j][:] = compute_Cjk(X, index, j, neighbors)

        return C

    def compute_wij(C_inv):

        wij = np.zeros((1, nneighbors))
        overallSum = 0

        for i in range(nneighbors):
            for k in C_inv[i]:
                wij[0, i] += k
                overallSum += k

        wij = np.multiply(1. / overallSum, wij)

        return wij

    def assign_Weights(Weights, i, neighbors, wij):

        for j in range(nneighbors):
            Weights[i, neighbors[j]] = wij[0, j]

        return Weights

    for i in range(X.shape[0]):
        neighbors = correctNeighbors[i]

        C = compute_C(X, i, neighbors)
        C_reg = C + np.multiply(np.trace(C) / 1000, np.identity(nneighbors))

        C_inv = np.linalg.inv(C_reg)

        wij = compute_wij(C_inv)

        Weights = assign_Weights(Weights, i, neighbors, wij)

    weight_end = datetime.now()

    delta_weight = weight_end - weight_start

    print("Duration of Weight Computation: ", delta_weight)

    ########################################
    # Checking of Sum in Weights
    ########################################

    check_sum = 0

    for i in Weights:
        for j in i:
            check_sum += j

    if check_sum is not float(nfeatures):
        print("Something went wrong!")
        print("Overall Test Sum is: ", check_sum)

    ########################################
    # Building Matrix M
    ########################################

    M_Build_start = datetime.now()

    I = np.identity(nfeatures)

    WI = (I - Weights)
    M = WI.transpose().dot(WI)

    M_Build_end = datetime.now()

    delta_M_Build = M_Build_end - M_Build_start

    print("Duration of M Computation: ", delta_M_Build)

    ########################################
    # Computation of u and y
    ########################################

    eigenvector_start = datetime.now()

    eigen_values, eigen_vectors = linalg.eigh(M, eigvals=(1, ncomponents + 1 - 1), overwrite_a=True)
    index = np.argsort(np.abs(eigen_values))
    y = eigen_vectors[:, index]

    eigenvector_end = datetime.now()

    delta_eigenvector = eigenvector_end - eigenvector_start

    print("Duration of Computation of Eigenvectors: ", delta_eigenvector)

    ########################################
    # Generating Output
    ########################################

    return y






