import numpy as np
import cv2
from pathlib import Path

def resize_images(input_file, output_file, new_shape):
    # Load the images
    images = np.load(input_file)

    # Resize the images
    resized_images = []


    for image in images:
            # Resize the image using OpenCV
        resized_image = cv2.resize(image[0], (new_shape[1], new_shape[0]))

        # Append resized image to list
        resized_images.append(resized_image)

    # Convert list of resized images to numpy array
    resized_images = np.array(resized_images)

    # Save the resized images
    np.save(output_file, resized_images)

if __name__ == "__main__":
    # Define the input and output file paths
    input_train_file = Path("X_train.npy")
    input_test_file = Path("X_test.npy")
    output_train_file = Path("resized_X_train1.npy")
    output_test_file = Path("resized_X_test1.npy")

    # Define the new shape for resizing
    new_shape = (264, 264)

    # Resize the first 10 images from the training dataset
    resize_images(input_train_file, output_train_file, new_shape)

    # Resize the first 10 images from the testing dataset
    resize_images(input_test_file, output_test_file, new_shape)
