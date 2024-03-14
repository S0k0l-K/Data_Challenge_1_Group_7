from torch.utils.data import WeightedRandomSampler
# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
from collections import Counter
import numpy as np
import os
from pathlib import Path

current_script_path = Path(__file__).parent.parent

# Now, construct the paths to your dataset files correctly
train_x_path = current_script_path / "data" / "X_train.npy"
train_y_path = current_script_path / "data" / "Y_train.npy"
test_x_path = current_script_path / "data" / "X_test.npy"
test_y_path = current_script_path / "data" / "Y_test.npy"

# Assuming you have the appropriate constructor for ImageDataset, use the paths
train_dataset = ImageDataset(train_x_path, train_y_path)
test_dataset = ImageDataset(test_x_path, test_y_path)

# Function to calculate class counts
def calculate_class_sample_counts(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)
    class_counts = Counter(labels)
    class_sample_counts = [class_counts[i] for i in range(len(class_counts))]
    return class_sample_counts

# Calculate class sample counts for the training dataset
train_class_sample_counts = calculate_class_sample_counts(train_dataset)
print("Training class sample counts:", train_class_sample_counts)

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dc1.image_dataset import ImageDataset  # Ensure you have this or similar for loading your data
import argparse
from pathlib import Path


def main(args):
    # Load the train and test dataset
    train_dataset = ImageDataset(Path("X_train.npy"), Path("Y_train.npy"))
    test_dataset = ImageDataset(Path("X_test.npy"), Path("Y_test.npy"))


    # Extract labels directly from the train_dataset.targets attribute
    train_labels = train_dataset.targets  # Now directly accessing the labels

    # Assuming these are your class sample counts
    class_sample_counts = [2521, 2318, 2964, 6103, 1633, 1302]
    total_samples = sum(class_sample_counts)
    class_weights = [total_samples / count for count in class_sample_counts]

    # Generate a list of weights for each sample in the dataset, based on the class they belong to
    sample_weights = [class_weights[label] for label in train_labels]

    # Create a sampler for your dataset. This will be used in the DataLoader
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoader for training, now using the sampler for balanced classes
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    # DataLoader for testing can remain unchanged as we typically don't need to balance classes for evaluation
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Your model training and testing code goes here...
...