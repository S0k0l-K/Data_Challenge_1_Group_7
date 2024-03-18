import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def load_data_and_create_dataset(x_path, y_path):
    base_path = Path(__file__).parent
    x_path = base_path / x_path  
    y_path = base_path / y_path

    x = np.load(x_path)  
    y = np.load(y_path)

    print("Loaded x shape:", x.shape)  

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)

    return dataset


class CustomResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_path = Path(__file__).parent.parent / "data" / "X_train.npy"
    y_path = Path(__file__).parent.parent / "data" / "Y_train.npy"

    train_dataset = load_data_and_create_dataset(x_path, y_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CustomResNet(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    torch.save(model.state_dict(), 'resnet_model.pth')
    print("Finished Training and saved the model")


if __name__ == "__main__":
    main()
