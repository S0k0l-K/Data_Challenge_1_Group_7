import numpy as np
import matplotlib.pyplot as plt
#Train data, rotated trial data and zoomed trial data loaded
test = np.load('X_train.npy')
rotated_trial = np.load('augmented_subset_images_trial.npy')
zoomed_trial = np.load('zoomed_subset_images_trial.npy')
# You can check the shape of the dataset from the commented line below
# print(test.shape)

# from PIL import Image

# Code for displaying individual training images
# image_to_display = np.squeeze(test[3])
#
# plt.imshow(image_to_display, cmap='gray')
# plt.show()

# Code for displaying individual rotated images
# image_to_display = np.squeeze(rotated_trial[4])
#
# plt.imshow(image_to_display, cmap='gray')
# plt.show()

# Code for displaying individual zoomed images
# image_to_display = np.squeeze(zoomed_trial[4])
#
# plt.imshow(image_to_display, cmap='gray')
# plt.show()


