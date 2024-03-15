import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path  # Add this line

# Assuming dc1 modules are correctly implemented and available
from dc1.image_dataset import ImageDataset

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

# Replicate the CustomResNet model structure from ResNetSt.py
class CustomResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    # Modify target names as per your dataset
    target_names = ['Pneumothorax', 'Nodule', 'Infiltration', 'Effusion', 'Atelectasis', 'No Finding']
    print(classification_report(all_labels, all_preds, target_names=target_names))

def main():
    # Adjust paths as necessary
    # Path adjustments for dataset
    data_path = Path(".")
    train_x_path = data_path / "X_train.npy"
    train_y_path = data_path / "Y_train.npy"
    test_x_path = data_path / "X_test.npy"
    test_y_path = data_path / "Y_test.npy"
    test_x_path = Path(__file__).parent.parent / "data" / "X_test.npy"
    test_y_path = Path(__file__).parent.parent / "data" / "Y_test.npy"

    test_dataset = ImageDataset(test_x_path, test_y_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=6).to(device)

    # Load the trained model
    model.load_state_dict(torch.load('custom_resnet_model.pth', map_location=device))

    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
