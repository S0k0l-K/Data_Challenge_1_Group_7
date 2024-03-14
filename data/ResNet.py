import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from pathlib import Path
from collections import Counter


def load_data_and_create_dataset(x_path, y_path):
    base_path = Path(__file__).parent
    x_path = base_path / x_path  # Correct path construction
    y_path = base_path / y_path

    x = np.load(x_path)  # Load the data correctly
    y = np.load(y_path)

    # No need to reshape or add a channel dimension since the data is already in the correct shape
    print("Loaded x shape:", x.shape)  # Debug: Check the shape

    # Convert numpy arrays to PyTorch tensors, ensuring the correct type
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)

    return dataset


class CustomResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        # Adjust the first Conv2d layer to accept 1 channel input for grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Modify paths as needed
    x_path = Path(__file__).parent.parent / "data" / "X_train.npy"
    y_path = Path(__file__).parent.parent / "data" / "Y_train.npy"

    # Load the dataset
    train_dataset = load_data_and_create_dataset(x_path, y_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CustomResNet(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'resnet_model.pth')
    print("Finished Training and saved the model")




if __name__ == "__main__":
    main()
