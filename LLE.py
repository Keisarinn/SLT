
import numpy as np
import matplotlib.pyplot as plt
import os.path
from time import time
import copy
from mnist_data import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors

def plot_embedding(X, y, component, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    if component == 2:
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.TextArea(y[i]),
                    X[i])
                ax.add_artist(imagebox)
    elif component == 3:
        ax = plt.subplot(111, projection='3d')
        for i in range(X.shape[0]):
            ax.text(X[i, 0], X[i, 1], X[i,2], str(y[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def vectorize_MNIST(X_train, X_test):
    num_pixels = X_train.shape[2] * X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    # normalize inputs from 0-255 to 0-1
    #X_train = X_train / 255
    #X_test = X_test / 255
    return X_train, X_test

def LLE(data, n_neighbors, component, number):
    data = data[0:number,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto', p=2).fit(data)
    neighbors_exact = nbrs.kneighbors(data, return_distance=False)
    neighbors_exact = neighbors_exact[:,1:]
    W = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        neighbors = neighbors_exact[i,:]
        C_matrix = np.zeros((neighbors.shape[0],neighbors.shape[0]))
        for j in range(neighbors.shape[0]):
            for k in range(neighbors.shape[0]):
                C_matrix[j,k] = np.dot(data[i,:]-data[neighbors[j],:],data[i,:]-data[neighbors[k],:])
        C_matrix = np.linalg.pinv(C_matrix)
        for j in range(neighbors.shape[0]):
            W[i,neighbors[j]] = C_matrix[j,:].sum()/C_matrix.sum()
    I = np.identity(W.shape[0])
    M = np.dot((I-W),(I-W))
    return W, M


X_train, y_train, X_test, y_test = load_dataset()
X_train, X_test = vectorize_MNIST(X_train, X_test)
number = 800;
component = 3;
W, M = LLE(X_test, 50, component, number)
a,b =  np.linalg.eig(M)
x = np.arange(a.shape[0])
X_testt = X_test[0:number,:]
X_lle = np.zeros(X_testt.shape)
for i in range(number):
    for j in range(number):
        X_lle[i,:] = W[i,j]*X_testt[j,:]+ X_lle[i,:]
plot_embedding(X_lle, y_test[:number], component)
plt.figure()
plt.plot(x, np.sort(np.absolute(a)),'ro')
plt.show()
a = 0;