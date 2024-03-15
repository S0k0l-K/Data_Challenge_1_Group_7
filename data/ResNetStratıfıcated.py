import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import models
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import argparse

# Assuming dc1 modules are correctly implemented and available in the environment
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model

# Function to calculate class counts
def calculate_class_sample_counts(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)
    class_counts = Counter(labels)
    class_sample_counts = [class_counts[i] for i in range(len(class_counts))]
    return class_sample_counts

# Custom ResNet model for our specific use case
class CustomResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        # Adjust to accept 1 channel input if necessary, or adjust according to your dataset
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def main():
    # Path adjustments for dataset
    data_path = Path(".")
    train_x_path = data_path / "X_train.npy"
    train_y_path = data_path / "Y_train.npy"
    test_x_path = data_path / "X_test.npy"
    test_y_path = data_path / "Y_test.npy"

    # Loading datasets
    train_dataset = ImageDataset(train_x_path, train_y_path)
    test_dataset = ImageDataset(test_x_path, test_y_path)

    # Calculate class sample counts for the training dataset
    train_class_sample_counts = calculate_class_sample_counts(train_dataset)

    # Assuming class sample counts are correct, proceed with weighted sampling
    total_samples = sum(train_class_sample_counts)
    class_weights = [total_samples / count for count in train_class_sample_counts]
    train_labels = [label for _, label in train_dataset]
    sample_weights = [class_weights[label] for label in train_labels]

    # Create a sampler and DataLoader for balanced class distribution
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=len(class_weights)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (simplified for demonstration)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)  # Convert labels to LongTensor
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Save the model after training
    torch.save(model.state_dict(), 'custom_resnet_model.pth')

if __name__ == "__main__":
    main()
