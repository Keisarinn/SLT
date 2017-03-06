import loader as load
import viewer as view

import numpy as np
import sklearn.manifold as manifold
import lle as lle
import scipy.linalg as linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mnist_images,mnist_labels = load.get_labeled_data("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
mnist_matrix=load.to_feature_matrix(mnist_images)

training_images=mnist_matrix[0:1000,:]


def plot_2D_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
 
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
def plot_3D_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],X[i,2], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
result,M_matrix=lle.lle_algorithm(X=training_images, n_neighbours=50, y_dim=2)

#EW, EV=linalg.eigh(M_matrix, eigvals_only=False, turbo=True)
#plt.plot(EW)
#plt.savefig('EW.eps')

#slt = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=2, neighbors_algorithm='kd_tree')
#result = slt.fit_transform(training_images)

plot_2D_embedding(X=result, y=mnist_labels[0:1000, 0])
#view.view_image(M_matrix)
plt.savefig('images2.eps')

#%% Linear manifold interpolation
from pylab import imshow, cm
x1=mnist_images[0,:,:]
x2=mnist_images[1,:,:]
images_interp1=0.5*x1+0.5*x2
images_interp2=0.75*x1+0.25*x2
images_interp3=0.25*x1+0.75*x2

imshow(x1, cmap=cm.gray)
plt.savefig('images_interp1.eps')
imshow(x2, cmap=cm.gray)
plt.savefig('images_interp5.eps')
imshow(images_interp1, cmap=cm.gray)
plt.savefig('images_interp3.eps')
imshow(images_interp2, cmap=cm.gray)
plt.savefig('images_interp2.eps')
imshow(images_interp3, cmap=cm.gray)
plt.savefig('images_interp4.eps')

#%% Reconstruction
import numpy as np
import scipy.spatial as spatial
from numpy.linalg import inv
import scipy.linalg as linalg

from pylab import imshow, cm

new_data=result[0,:]*0.25+result[1,:]*0.75
result_new=result
result_new[999,:]=new_data

n_neighbours=5
data_num=999

ckd_tree=spatial.cKDTree(data=result, leafsize=100)
d, NN_self=ckd_tree.query(result[data_num],k=n_neighbours+1,p=1)
NN=np.delete(NN_self,0)

C=np.zeros([n_neighbours,n_neighbours])
nn_matrix=result[NN,:]

def w_numerator(C_inv,j):
    numerator=0
    for i in range(C_inv.shape[0]):
        numerator=numerator+C_inv[j][i]
    return numerator

def w_denominator(C_inv):
    denominator=0
    for l in range(C_inv.shape[0]):
        for k in range(C_inv.shape[0]):
            denominator=denominator+C_inv[l][k]
    return denominator
        
for j in range(n_neighbours):
    for k in range(n_neighbours):
        a=result[data_num,:]-nn_matrix[j]
        b=result[data_num,:]-nn_matrix[k]
        C[j][k]=np.dot(a,b.T)

C_inv=inv(C)
        
W_vector=np.zeros([1,n_neighbours])
        
for index_j in range(n_neighbours):
    W_vector[0][index_j]=w_numerator(C_inv,index_j)/w_denominator(C_inv)
    
image_reconstructed=np.zeros([28,28])
for index in range(n_neighbours):
    image_reconstructed=image_reconstructed + W_vector[0,index]*mnist_images[NN[index],:,:]


imshow(image_reconstructed, cmap=cm.gray)
plt.savefig('images_reconstructed2.eps')










