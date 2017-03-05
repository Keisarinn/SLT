from __future__ import division
import numpy as np
import numpy.matlib as matlib
import scipy.linalg as linalg

def giac_lee(X, K, d):
    """ X: Matrix with points as COLUMNS
    K: number of nearest neighbours
    d: dimension to reduce to
    """
    D = X.shape[0]
    N = X.shape[1]
    print K
    print N

    # Find nearest neighbours
    X_square = np.square(X)
    X2 = X_square.sum(axis=0)
    distance = matlib.repmat(X2, N, 1) - np.multiply(2, np.dot(X.T, X))
    print "distance shape: " + str(distance.shape)
    index = np.argsort(distance)
    neighborhood = index[1:(1+K), :]
    print "neighborhood: " + str(neighborhood.shape)

    # Solve reconstruction weights
    W = np.zeros((K,N))
    for ii in range(N):
        cur_x = np.matrix(X[:,ii])
        print "cur x: " + str(cur_x.shape)
        cur_neigh = X[:, neighborhood[:,ii]]
        print "cur neigh: " + str(cur_neigh.shape)
        cur_neigh = cur_neigh[:,:,0]

        z = np.subtract(cur_neigh, np.repeat(cur_x,K,axis=1))
        print "z shape: " + str(z.shape)

        C = np.dot(z.T, z)
        reg = np.trace(C) / 1000 # Smallnum
        C = np.add(C, np.multiply(np.eye(K,K), reg))
        b = np.ones((K,1))
        print "W ii shape: " + str(W[:,ii].shape)
        W[:,ii] = linalg.solve(C, b).flatten()
        W[:,ii] = W[:,ii]/np.sum(W[:,ii])

    # Embedding from eigenvectors
    print "----------------------------------------------------"
    M = np.eye(N,N)
    for ii in range(N):
        w = np.matrix(W[:, ii])
        jj = neighborhood[:,ii]
        print "M i j: " + str(M[ii,jj].shape)
        print "M j i: " + str(M[jj,ii].shape)
        print "w: " + str(w.shape)
        M[ii,jj] = np.subtract(M[ii,jj], w.T)
        M[jj,ii] = np.subtract(M[jj,ii], w.T)
        M[jj,jj] = np.dot(w,w.T)

    # Embedding
    eigenvals, eigenvectors = np.linalg.eigh(M)
    idx = eigenvals.argsort()
    sorted_eigen = eigenvectors[:,idx]
    print eigenvals[idx]
    Y = np.multiply(sorted_eigen[:, 1:d+1].T, np.sqrt(N))
    return Y

def fit_LLE(X, n_neighbours, n_components):
    Y = giac_lee(X.T, n_neighbours, n_components)
    return Y.T
