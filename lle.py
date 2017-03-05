import numpy as np
from sklearn.manifold.locally_linear import barycenter_kneighbors_graph
from sklearn.manifold.locally_linear import null_space
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import eye, csr_matrix


#Code from sklearn
def locally_linear_embedding(
        X, n_neighbors, n_components, metric = 'euclidean', reg=1e-3, eigen_solver='auto', tol=1e-6,
        max_iter=100, method='standard', hessian_tol=1E-4, modified_tol=1E-12,
        random_state=None, n_jobs=1):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs.fit(X)
    X = nbrs._fit_X

    W = barycenter_kneighbors_graph(nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)

    # we'll compute M = (I-W)'(I-W)
    M = eye(*W.shape, format=W.format) - W
    M = (M.T * M).tocsr()

    return (null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                       tol=tol, max_iter=max_iter, random_state=random_state), M)
