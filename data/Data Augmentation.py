from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Image generator of flipped and rotated imag

x_train = np.load('X_train.npy')
x_train = x_train.reshape((16841, 128, 128, 1))
subset_images = x_train
# batch_size = 40

data_generator = ImageDataGenerator(rotation_range=15)
augmented_images_batch = next(data_generator.flow(subset_images,batch_size=len(subset_images), seed=3213))


zoom_generator = ImageDataGenerator(zoom_range=0.1) #Zoom in or out by 10%
x_train = np.load('X_train.npy')
x_train = x_train.reshape((16841, 128, 128, 1))
subset_images = x_train



zoomed_images_batch = next(zoom_generator.flow(subset_images,batch_size=len(subset_images), seed=3213))

np.save('zoomed_subset_images_trial.npy', zoomed_images_batch)
np.save('augmented_subset_images_trial.npy', augmented_images_batch)



