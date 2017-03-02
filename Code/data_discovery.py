import os
from mnist import MNIST
import numpy as np
from numpy import unravel_index


#Set up paths
__file__ = "/data/train-images.idx3-ubyte"
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
curr_cwd = os.getcwd()
# train_file = "data/train-images.idx3-ubyte"
# __location__ = os.path.join(curr_cwd, train_file)

mndata = MNIST(os.path.join(curr_cwd,"data"))
images, labels = mndata.load_training()
images = np.asarray(images[:200])
labels = np.asarray(labels[:200])

import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from sklearn import manifold

def LLE_sk(components = 2, neighbors = 4):
    X_r, err = manifold.locally_linear_embedding(images, n_neighbors=neighbors,
                                                    n_components=components)

    ax = plt.subplot(212, projection='3d')
    ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=labels, cmap=plt.cm.Spectral)
    ax.set_aspect('auto')
    plt.axis('tight')
    # plt.xticks([]), plt.yticks([])
    plt.title('LLE 3D')
    ax = plt.subplot(211)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_title("LLE 2D")
    ax.set_aspect('auto')
    plt.show()


def LLE(k = 4):

    n = len(images)
    D = images.shape[1]
    distances = np.zeros((n,n))

    # 1. Calculate distances and choose k nearest neighbors
    for i in range(n):
        point = images[i]
        for j in range(n):
            other_point = images[j]
            # diff = abs(point - other_point)
            # distance = np.sum(diff)
            # distances[i,j] = distance
            euclidean_distance = np.linalg.norm(point - other_point)
            distances[i, j] = euclidean_distance

    # Select k closest neighbors
    distances_min = np.zeros((n,k))
    distances_min_index = np.zeros((n,k))
    for i in range(n):
        it = 0
        for j in [x for x in range(n) if x != i]:
            if it < k:
                distances_min[i][it] = abs(distances[i][j])
                distances_min_index[i][it] = j
                it+=1
            else:
                max_distance = np.max(distances_min)
                if distances[i][j] < max_distance:
                    max_index = unravel_index(distances_min[i].argmax(),distances_min[i].shape)
                    distances_min_index[i][max_index] = j #Update index for distance
                    new_dist = distances[i][j]
                    distances_min[i][max_index] = new_dist # Change the distance


    # We now have nearest neighbors for every point in distances_min_index

    #Build up the Z matrix, subtract X_i
    Z_index =   [[int(j) for j in x ]for x in distances_min_index]
    Z = [[[0]*images.shape[1] for x in range(k)] for i in range(n)]
    for i in range(n):
        it = 0
        for j in Z_index[i]:
            Z[i][it] = images[j]
            it+=1
        # Z[i] = Z[i] - images[i]

    # Should be centered around origo

    # 2. Solve for the reconstruction weights W
    Z = np.asarray(Z)
    # W = np.asarray([[[[0] for l in range(k)] for j in range(n)] for i in range(n)])
    W = np.zeros((n,n))
    for i in range(n):
        Z_i = Z[i,:,:]
        Z_i = Z_i - images[i]
        C_i = np.dot(Z_i, Z_i.T)
        #Fucking singular shit matrix C_i, please help
     #   C_i = C_i + np.identity(784)*0.2 # Add a little bit of identity matrix to make the matrix non-singular. Random adding.
        ones = np.ones((4,1))
        W_i = np.linalg.solve(C_i, ones)  #This takes time
        W_i = W_i/sum(W_i)
        # Place these weights at the right place in the weight matrix, W!
        neighbors_index = Z_index[i]
        for j in range(k):
            neighbor_index = neighbors_index[j]
            print("Neighbor index: {} and weight: {}".format(neighbor_index,W_i[j]))
            W[i,neighbor_index] = W_i[j,0]

    # 2nd done!

    # 3. Compute embedding coordinates Y using weights W.



    # W = [[[0] for l in range(k)] for i in range(n)]
    # for i in range(n):
    #     Z_i = Z[i, :, :]
    #     Z_i = Z_i - images[i]
    #     for j in range(k):
    #         neighbor = Z_i[j,:]
    #         C_i_j = [[[0] for y in range(D)] for x in range(D)]
    #         for l in range(k):
    #             other_neighbor = Z_i[l,:]
    #             C_i_j += np.outer(neighbor,other_neighbor)
    #         #Covariance matrix created for this neighbor
    #         #Solve for the weights!
    #         ones = np.ones((784, 1))
    #         C_i_j = np.array(C_i_j)
    #         W_i_j = np.linalg.solve(C_i_j, ones)  # This takes time
    #     a = 2

        # C_i = np.dot(Z_i.T, Z_i)
        # # Fucking singular shit matrix C_i, please help
        # C_i = C_i + np.identity(
        #     784) * 0.2  # Add a little bit of identity matrix to make the matrix non-singular. Random adding.
        # ones = np.ones((784, 1))
        # W_i = np.linalg.solve(C_i, ones)  # This takes time
        # W_i = W_i / sum(W_i)
        # # Place these weights at the right place in the weight matrix, W!
        # neighbors = Z_index[i]



LLE(4)


## Code of ze forgotten!

# #Calculating the Covariance matrix
# C = [[[[0] for l in range(k)] for j in range(k)] for i in range(n)]
# for i in range(n):
#     X_i = images[i]
#     for j in range(k):
#         X_j = Z[i][j]
#         for l in range(k):
#             X_l = Z[i][l]
#             C[i][j][l] = np.dot((X_i-X_j),(X_i-X_l))