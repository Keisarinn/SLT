import numpy as np
import scipy.spatial as spatial
from numpy.linalg import inv
import scipy.linalg as linalg

def lle_algorithm(X, n_neighbours, y_dim):
    W_matrix=np.zeros([X.shape[0],X.shape[0]])
    ckd_tree=spatial.cKDTree(data=X, leafsize=100)
    
    for index in range(X.shape[0]):
        d, NN_self=ckd_tree.query(X[index],k=n_neighbours+1,p=1)
        NN=np.delete(NN_self,0)
        
        C=np.zeros([n_neighbours,n_neighbours])
        nn_matrix=X[NN,:]
        for j in range(n_neighbours):
            for k in range(n_neighbours):
                a=X[index,:]-nn_matrix[j]
                b=X[index,:]-nn_matrix[k]
                C[j][k]=np.dot(a,b.T)
        C_inv=inv(C)
        
        for index_j in range(n_neighbours):
            ind=NN[index_j]
            W_matrix[index][ind]=w_numerator(C_inv,index_j)/w_denominator(C_inv)
     
     
    M_matrix=np.zeros([W_matrix.shape[0],W_matrix.shape[0]])
    diff=np.identity(W_matrix.shape[0])-W_matrix
    M_matrix=np.dot(diff.T, diff)
    EW, EV=linalg.eigh(M_matrix, eigvals_only=False, eigvals=(0,y_dim),turbo=True)
    EV_trunc=np.delete(EV,(0),axis=1)
    return EV_trunc, M_matrix


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
        