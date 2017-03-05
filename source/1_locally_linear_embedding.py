# -*- coding: utf-8 -*-
"""
SLT coding exercise #1
Locally Linear Embedding

Author: Nicolas Känzig, 12-916-615
"""

import numpy as np
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
from mnist import MNIST
from numpy import linalg
from numpy import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
#from matplotlib import offsetbox

def plot_embedding(emb, labels, k):
    
    colors = np.array(['b','g','r','c','m','y','k','grey','orange','salmon'])
    #l = np.array(['0','1','2','3','4','5','6','7','8','9'])
    c = colors[np.array(labels)]
    
    if emb.shape[1] == 2:
        fig = plt.figure()
        
        x = emb[:,0]
        y = emb[:,1]
        plt.scatter(x,y, c=c, s=15)
    
        plt.title('Embedding Vectors with k=' + str(k) + ' neighbors')
    
    if emb.shape[1] == 3:
        fig = plt.figure()
        p3d = fig.add_subplot(111, projection='3d')
        
        x = emb[:,0]
        y = emb[:,1]
        z = emb[:,2]
        p3d.scatter(x,y,z, c=c, s=10)
        plt.title('Embedding Vectors with k=' + str(k) + ' neighbors')
        

def LLE(X, n_neighbors, n_components, nr_images):
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                          method='standard')
    X_lle = clf.fit_transform(X[:nr_images])
    
    return X_lle


def reconstruct(image, X_lle, X, k):
   
    obj = neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean').fit(X_lle)
    dist, k_neighbors = obj.kneighbors(image)

    k_neighbors = k_neighbors[:, 1:][0]  # indices of the k neares4t neighbors

    '''Solve for weights'''
    myVecs = np.array(X_lle)[k_neighbors, :]

    w = linalg.lstsq(myVecs.T, image)[0]

    ''' Apply weights to originalSpace'''
    Xorig = np.dot(w.T, X[k_neighbors, :])

    return Xorig


def main(): 
    ## Load data
    mndata = MNIST('./dataset/')
    X, labels = mndata.load_training()
    X = np.array(X)
    
    n_neighbors = 10 # number of neighbors on the set
    n_components = 2 # dimension of the embedded space
    nr_images = 1000 # number of images to process
    
    
    ## Calculate LLE
    X_lle = LLE(X, n_neighbors, n_components, nr_images)
    
    ## Plot embedded space
    plot_embedding(X_lle, labels[:nr_images], n_neighbors)
    
   
    ## Reconstruction  
    
    # Select a random reference point
    random_index = 17
    img_to_reconstruct = X_lle[random_index] 
    
    reconst_img = reconstruct(img_to_reconstruct, X_lle, X, n_neighbors)    
    reconst_img = np.reshape(np.array(reconst_img), (28,28));

    orig_img = np.reshape(np.array(X[random_index]), (28,28));
    
    f, ax1 = plt.subplots(1,2)
    ax1[0].matshow(orig_img, cmap=plt.cm.gray)
    ax1[1].matshow(reconst_img, cmap=plt.cm.gray)
    
#    ax1[0].set_title('Original Image')
#    ax1[1].set_title('Reconstructed Image')


main()

    



