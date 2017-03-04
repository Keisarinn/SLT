import numpy as np



def nearest_neighbor(X,nneighbor,norm):
    neighbors = np.zeros((X.shape[0],nneighbor))

    for i in range(X.shape[0]):
        index = distance_norm(X,X[i],norm)
        index = index.transpose()
        neighbors[i] = index[0,1:nneighbor+1]

    return neighbors










def distance_norm(points,point_in_points,norm):
    differences = points - point_in_points

    size = np.shape(points)
    size = size[0]

    distances = np.zeros((size,1))

    if norm == 2:
        for i in range(size):
            for j in differences[i]:
                distances[i] += j

            distances[i] = np.sqrt(distances[i])

    index = np.argsort(distances)

    return index


