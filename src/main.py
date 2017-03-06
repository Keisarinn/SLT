from LLE_implementation import LLE_implementation
from LLE_plots import plot_embedding_2D,plot_embedding_3D,matrix_plot
import matplotlib.pyplot as plt
import mnist
import numpy as np


########################################
# Preferences
########################################

nfeatures = 1000
ncomponents = 2
nneighbors = 9
metric = ['manhattan', 'minkowski', 'euclidean']


computation = False
sampling = False
plotting = False



########################################
# Importing Data
########################################

# Loading Data from MNIST

mndata = mnist.MNIST('MNIST_data')
training_x, training_y = mndata.load_training()

X, labels = np.asarray(training_x[0:nfeatures]), np.asarray(training_y[0:nfeatures])

index = np.argsort(labels,0,'quicksort')


labels_sorted = np.zeros((1000,1))
X_sorted = np.zeros((1000,784))

for i in range(0,1000):
    X_sorted[i] = X[index[i]]
    labels_sorted[i] = labels[index[i]]


########################################
# Data Manipulation
########################################

# Format of X: --> (nfeatures,ncomponents)

#X, labels = np.asarray(training_x[0:nfeatures]), np.asarray(training_y[0:nfeatures])


########################################
# LLE Computation
########################################

if computation is True:
    if sampling is True:
        for i in range(2, 31):
            y,M = LLE_implementation(X, nfeatures, i, ncomponents,metric)
            file_name = "LLE_with_" + str(ncomponents) + "_components_and_" + str(i) + "_neighbors"
            np.save(file_name, y)

    if sampling is not True:
        y,M = LLE_implementation(X,nfeatures,nneighbors,ncomponents,metric)


########################################
# Loading Data/ Plotting
########################################

if plotting is True:
    for i in metric:
        for j in range(2, 4):
            for k in range(2, 31):
                file_name = "LLE_with_" + str(j) + "_components_and_" + str(k) + "_neighbors.npy"
                if j == 2:
                    y = np.load("Results/2D/" + i + "/" + file_name)
                    plot_embedding_2D(y, labels, i, j, k)
                if j == 3:
                    y = np.load("Results/3D/" + i + "/" + file_name)
                    plot_embedding_3D(y, labels, i, j, k)


y, M = LLE_implementation(X_sorted,nfeatures,nneighbors,ncomponents,metric[1])


plt.figure()
plt.imshow(M,cmap='Dark2')
plt.show()
plt.close()

u,s,v = np.linalg.svd(M)


fig = plt.figure()
plt.plot(s)
plt.show()





