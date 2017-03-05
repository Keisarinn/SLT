import sklearn.manifold as manifold
import mnist_data
import numpy as np 
import LLE
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def plot_2D(X, y, title=None):
    # X[:,0] := x coordinate of eigenvectors
    # X[:,1] := y coordinate of eigenvectors
    # y := cluster labels
    plt.figure()
    ax = plt.subplot(111)
    scatter = ax.scatter(X[:,0], X[:,1], c=y)
    plt.colorbar(scatter)

def plot_3D(X, y, title=None):
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
    plt.colorbar(scatter)

def plot_svd(x):
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x)

print 'getting started'
# load data
file_X = 'train-images.gz'
file_Y = 'train-labels.gz'

training_X, training_Y = mnist_data.get_labeled_data(file_X, file_Y)

### Linear Embedding implementation ###
print '2d case begins'
## 2D case with euclidean##
M_2D, lle_2D = LLE.fit_transform(X=training_X, n_neighbors=5, n_components=2)
plot_2D(lle_2D, training_Y)

print 'M_2d'
plt.matshow(M_2D[0:100, 0:100])

print 'svd_2d'
U, s_2D, V = np.linalg.svd(M_2D)
plot_svd(s_2D[0:100])
'''
## 3D case with euclidean##
print '3d case begins'
M_3D, lle_3D = LLE.fit_transform(X=training_X, n_neighbors=5, n_components=3)
plot_3D(lle_3D, training_Y) 


## 2D case with minkowski
print 'minkowski'
M_m2D, lle_m2D = LLE.fit_transform(X=training_X, n_neighbors=5, n_components=2, metric='minkowski')
plot_2D(lle_m2D, training_Y)

## 2D case with more neighbors
print 'more neighbors'
M_n2D, lle_n2D = LLE.fit_transform(X=training_X, n_neighbors=30, n_components=2)
plot_2D(lle_n2D, training_Y)

M_n3D, lle_n3D = LLE.fit_transform(X=training_X, n_neighbors=30, n_components=3)
plot_3D(lle_n3D, training_Y)

## 2D case with more neighbors and minkowski
print 'min + more'
M_n2D, lle_mn2D = LLE.fit_transform(X=training_X, n_neighbors=30, n_components=2, metric='minkowski')
plot_2D(lle_mn2D, training_Y)

M_n3D, lle_mn3D = LLE.fit_transform(X=training_X, n_neighbors=30, n_components=3, metric='minkowski')
plot_3D(lle_mn3D, training_Y)
'''

plt.show()