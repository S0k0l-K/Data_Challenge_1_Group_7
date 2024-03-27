import torch
from torchvision import transforms
import numpy as np
from pathlib import Path


data_path = Path(__file__).parent.parent
train_x_path = data_path / "data/X_train.npy"
train_y_path = data_path / "data/Y_train.npy"
test_x_path = data_path / "data/X_test.npy"
test_y_path = data_path / "data/Y_test.npy"

x_train, y_train = np.load(train_x_path), np.load(train_y_path)
x_test, y_test = np.load(test_x_path), np.load(test_y_path)


class_counts = [np.sum(y_train == i) for i in range(6)]
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
class_weights = class_weights / class_weights.sum()  # Normalize weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)


# Transform to convert numpy images to PyTorch tensors
to_tensor_transform = transforms.ToTensor()

# Variables to store total and squared pixel values
total_pixels = 0
sum_of_pixels = 0.0
sum_of_pixels_squared = 0.0

for image in x_train:
    # Convert image to tensor and scale pixel values to [0, 1]
    tensor_image = to_tensor_transform(image).float()

    # Accumulate the sum of pixel values and their squares
    sum_of_pixels += tensor_image.sum()
    sum_of_pixels_squared += (tensor_image ** 2).sum()

    # Keep track of the total number of pixels
    total_pixels += tensor_image.numel()

# Calculate mean and std
mean = sum_of_pixels / total_pixels
variance = (sum_of_pixels_squared / total_pixels) - (mean ** 2)
std = variance ** 0.5

print(f"Mean: {mean}, Std: {std}")
