import numpy as np
import matplotlib.pyplot as plt
test = np.load('X_test.npy')
print(test.shape)
from PIL import Image

image_to_display = np.squeeze(test[1])

plt.imshow(image_to_display, cmap='gray')
plt.show()

