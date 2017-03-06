import numpy as np
import mnist
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


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

    return Weights, correctNeighbors


ncomponents = 3
nneighbors = 9

mndata = mnist.MNIST('MNIST_data')
training_x, training_y = mndata.load_training()

X, labels = np.asarray(training_x[0:5000]), np.asarray(training_y[0:5000])



file_name = "LLE_with_" + str(ncomponents) + "_components_and_" + str(nneighbors) + "_neighbors.npy"

if ncomponents == 2:
    y = np.load("Results/2D/euclidean/" + file_name)
if ncomponents == 3:
    y = np.load("Results/3D/euclidean/" + file_name)



Weights, neighbors = LLE_implementation(y,5000,nneighbors,ncomponents,'euclidean')

print(np.shape(X))

x_new = np.zeros((1,784))

for i in range(5000):
    x_new += Weights[4][i]*X[i]

img = np.reshape(x_new,(28,28))

plt.figure()
plt.imshow(img,'gray_r')
plt.show()

print('Done!')