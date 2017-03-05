import new_lle as lle
import mnist_access as ma
import proj_one_constants as const
import matplotlib.pyplot as plt
import numpy as np
import plot_util as util
import sklearn.manifold as manifold

# Load image data and use a subset
test_images, test_labels = ma.get_labeled_data(const.TEST_IM, const.TEST_LABELS)
print test_labels.shape
test_matrix = ma.to_feature_matrix(test_images)
RESTRIC_NUM = 1000
test_restrict = test_matrix[0:RESTRIC_NUM,:]

# my LLE
NN = 5  # Number of neighbours
NC = 2  # Number of components
emb = lle.fit_LLE(test_restrict, NN, NC)

# util.plot_components(emb, test_labels)

i = 0
# for x in range(-20, 20, 2):
#     new_point = np.array([x, -1])
#     new_x = lle.interpolation(new_point, emb, test_images, NN)
#     print new_x.shape
#     plt.figure()
#     plt.imshow(new_x, cmap='Spectral')
#     plt.imsave("interp/im_"+str(i) + ".png", new_x, format='png')
#     i = i+1

im_start = np.reshape(a=test_images[0], newshape=(28,28))
im_end = np.reshape(a=test_images[1], newshape=(28,28))

plt.imshow(im_end)
plt.show()

lle.interp_original_space(im_start, im_end, 10)
