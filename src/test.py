import mnist
import numpy as np

from nearestNeighbor_implementation import nearest_neighbor


########################################
# Importing Data
########################################

# Loading Data from MNIST

mndata = mnist.MNIST('MNIST_data')
training_x, training_y = mndata.load_training()



########################################
# Data Manipulation
########################################

# Format of X: --> (nfeatures,ncomponents)

nfeatures = 2000

X, labels = np.asarray(training_x[0:nfeatures]), np.asarray(training_y[0:nfeatures])

neighbors = nearest_neighbor(X,5,2)

print(neighbors)