import numpy as np
import numpy
import idx2numpy
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from IPython.core.debugger import Tracer


X_train = idx2numpy.convert_from_file('/Users/phoebeliu/Downloads/slt_pro_1/train-images-idx3-ubyte')
Y_train = idx2numpy.convert_from_file('/Users/phoebeliu/Downloads/slt_pro_1/train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file('/Users/phoebeliu/Downloads/slt_pro_1/t10k-images-idx3-ubyte')
Y_test = idx2numpy.convert_from_file('/Users/phoebeliu/Downloads/slt_pro_1/t10k-labels-idx1-ubyte')

num_pixels = X_train.shape[1] * X_train.shape[2]
x_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
x_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
x_train = x_train[0: 1000, :]
y_train = Y_train[0: 1000]
n_neighbors = 5

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y_train[i]), color=plt.cm.Set1(y_train[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]]) 
        for i in range(x_train.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
             #   continue
            shown_images = np.r_[shown_images, [X[i]]]
            #imagebox = offsetbox.AnnotationBbox(
             #   offsetbox.OffsetImage(X_train[i, :, :], cmap=plt.cm.gray_r), X[i])
            #ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    fig.savefig('test_fig_%i_neighbors_imp.png' %(n_neighbors))

def plot_embedding3D(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(y_train[i]), color=plt.cm.Set1(y_train[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1., 1.]]) 
        for i in range(x_train.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
             #   continue
            shown_images = np.r_[shown_images, [X[i]]]
            #imagebox = offsetbox.AnnotationBbox(
             #   offsetbox.OffsetImage(X_train[i, :, :], cmap=plt.cm.gray_r), X[i])
            #ax.add_artist(imagebox)
    if title is not None:
        plt.title(title)
    fig.savefig('test3D_fig_%i_neighbors_imp.png' %(n_neighbors))

def plot_M(M):
	fig = plt.figure()
	plt.imshow(M)
	fig.savefig('M_fig_%i_neighbors.png' %(n_neighbors))

def plot_3DM(M):
	fig = plt.figure()
	plt.imshow(M)
	fig.savefig('M3D_fig_%i_neighbors.png' %(n_neighbors))

def LLE(X, n_neighbors, n_components):
	N, m = X.shape
	W = numpy.zeros((N,N))
	nbrs = NearestNeighbors(n_neighbors = n_neighbors, algorithm = 'auto').fit(X)
	k_nn = nbrs.kneighbors(X, return_distance = False)
	for row in range(N):
		knn = X[k_nn[row, :]]
		knn = knn[1:, :]
		neighbors = numpy.zeros((len(knn), 784))
		for i in range(len(knn)):
			neighbors[i, :] = numpy.array(knn[i, :] - X[row, :])
		C = numpy.zeros((len(neighbors), len(neighbors)))
		for i in range(len(knn)):
			for j in range(len(knn)):
				C[i, j] = np.dot(neighbors[i, :], neighbors[j, :])
		C_inv = np.linalg.inv(C)
		sum = np.sum(C_inv)
		for k in range(len(C)):
			W[row, k_nn[row, k]] = np.sum(C_inv[k, :]) / sum
	M = np.dot((numpy.identity(len(W)) - W).transpose(), numpy.identity(len(W)) - W)
	eig_vals, Y = eigh(M, eigvals = (1, n_components))
	return Y, M

X_lle, M = LLE(x_train, n_neighbors, n_components=2)
print X_lle.shape
#print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle, "Locally Linear Embedding of the digits (N-neighbors %i)" %(n_neighbors))
plot_M(M)

X_lle3, M3 = LLE(x_train, n_neighbors, n_components=3)
print X_lle3.shape
#print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding3D(X_lle3, "Locally Linear Embedding of the digits (N-neighbors %i)" %(n_neighbors))
plot_3DM(M3)
