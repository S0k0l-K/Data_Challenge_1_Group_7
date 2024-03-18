import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import models
from pathlib import Path
from collections import Counter


class CustomResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def load_augmented_data(x_path, y_path):
    x = np.load(x_path)
    y = np.load(y_path)
    y = np.repeat(y, 2)
    return x, y

def prepare_dataset(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32).transpose(1,3).transpose(2,3)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)

def calculate_weights_for_balanced_classes(y, num_classes):
    count = [0] * num_classes
    for label in y:
        count[label] += 1
    weight_per_class = [0.] * num_classes
    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(y)
    for idx, val in enumerate(y):
        weight[idx] = weight_per_class[val]
    return weight

def main():
    data_path = Path(".")
    augmented_x_path = data_path / "combined_subset_images.npy"
    train_y_path = data_path / "Y_train.npy"

    x_augmented, y_augmented = load_augmented_data(str(augmented_x_path), str(train_y_path))
    train_dataset = prepare_dataset(x_augmented, y_augmented)

    weights = calculate_weights_for_balanced_classes(y_augmented, num_classes=6)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    torch.save(model.state_dict(), 'custom_resnet_model_augmented.pth')

if __name__ == "__main__":
    main()
