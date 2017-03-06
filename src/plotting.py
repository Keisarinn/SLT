import imageio

import mnist
import numpy as np
import matplotlib.pyplot as plt


########################################
# Load Data
########################################

mndata = mnist.MNIST('MNIST_data')
training_x, training_y = mndata.load_training()


########################################
# Plot Data
########################################

arr = np.zeros((28,28))

for i in range(28):
    for j in range(28):
        arr[i][j] = training_x[4][28*i + j]



plt.imshow(arr,cmap='gray_r')
plt.show()