from LLE_implementation import LLE_implementation
from LLE_plots import plot_embedding_2D,plot_embedding_3D,matrix_plot
import mnist
import numpy as np


########################################
# Preferences
########################################

nfeatures = 60000
ncomponents = 2
nneighbors = 15
metric = 'euclidean'


computation = True
sampling = True



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

X, labels = np.asarray(training_x[0:nfeatures]), np.asarray(training_y[0:nfeatures])


########################################
# LLE Computation
########################################

if computation is True:
    if sampling is True:
        for i in range(2, 20):
            y = LLE_implementation(X, nfeatures, i, ncomponents,metric)
            file_name = "LLE_with_" + str(ncomponents) + "_components_and_" + str(nneighbors) + "_neighbors"
            np.save(file_name, y)

    if sampling is not True:
        y = LLE_implementation(X,nfeatures,nneighbors,ncomponents,metric)


########################################
# Loading Data
########################################

if computation is not True:
    file_name = "LLE_with_" + str(ncomponents) + "_components_and_" + str(nneighbors) + "_neighbors.npy"
    y = np.load(file_name)



########################################
# Plotting
########################################

if ncomponents == 2:
    plot_embedding_2D(y, labels, title=None)

if ncomponents == 3:
    plot_embedding_3D(y, labels, title=None)

########################################
# Analysis of Matrix M
########################################



