import idx2numpy
from PIL import Image
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt


x_img = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
labels = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')

# img=Image.fromarray(x_img[0], 'L')
# img.show()

x = np.reshape(x_img, (10000,-1))[0:500,:]
labels = np.reshape(labels, (10000,-1))[0:500]

lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', neighbors_algorithm='auto', random_state=None, n_jobs=1)

y = lle.fit_transform(x)

mat = plt.scatter(y[:,0], y[:,1], c=labels)
#mat = plt.matshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)
plt.colorbar(mat, boundaries=np.arange(np.min(labels)-.5,np.max(labels)+1.5), ticks=np.arange(np.min(labels),np.max(labels)+1))
plt.show()
