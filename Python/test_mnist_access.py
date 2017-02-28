import mnist_access as ma
import proj_one_constants as const

all_im = ma.get_labeled_data(const.TEST_IM, const.TEST_LABELS)

feature = ma.to_feature_matrix(all_im[0])
print feature.shape


# print all_im[0]
# print all_im[0].shape
# print all_im[0][0].shape
#
# ma.view_image(all_im[0][0])
# ma.view_image(all_im[0][100])
# ma.view_image(all_im[0][1000])
