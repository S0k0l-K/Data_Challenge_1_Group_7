# hyperparameter.py
from mobilenetfinale import CustomDataset, FocalLoss, train_model
from torch.optim.lr_scheduler import StepLR
from dc1.train_test import train_model, test_model
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5113557577133179], std=[0.24621723592281342]),
])

# Make sure to define transform or import it if it's defined in mobilenetoptimized_loss_function.py
data_path = Path(__file__).parent.parent
train_x_path = data_path / "data/X_train.npy"
train_y_path = data_path / "data/Y_train.npy"
test_x_path = data_path / "data/X_test.npy"
test_y_path = data_path / "data/Y_test.npy"

x_train, y_train = np.load(train_x_path), np.load(train_y_path)
x_test, y_test = np.load(test_x_path), np.load(test_y_path)

# Initialize datasets
train_dataset = CustomDataset(x_train, y_train, transform=transform)
test_dataset = CustomDataset(x_test, y_test, transform=transform)

class_counts = [np.sum(y_train == i) for i in range(6)]
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
class_weights = class_weights / class_weights.sum()  # Normalize weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# DataLoader setup
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
momentums = [0.9, 0.95, 0.99]
weight_decays = [0, 1e-4, 1e-3]
step_sizes = [5, 10]  # Epochs after which to reduce learning rate
gamma_values = [0.1, 0.5]  # Multiplicative factor of learning rate decay
num_epochs = 10

# Keep track of the best hyperparameters and best validation accuracy
best_accuracy = 0
best_params = {}

# Perform grid search
for lr in learning_rates:
    for momentum in momentums:
        for weight_decay in weight_decays:
            for step_size in step_sizes:
                for gamma in gamma_values:
                    print(f"Testing hyperparameters: lr={lr}, momentum={momentum}, weight_decay={weight_decay}, step_size={step_size}, gamma={gamma}")

                    # Initialize the model
                    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
                    model = model.to(device)

                    # Define the loss function with class weights
                    criterion = FocalLoss(weight=class_weights, alpha=0.25, gamma=2.0, reduction='mean').to(device)
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

                    # Set up the learning rate scheduler
                    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

                    # Train the model using your training function
                    val_accuracy = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs)

                    # Update best hyperparameters if current model is better
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = {
                            'learning_rate': lr,
                            'momentum': momentum,
                            'weight_decay': weight_decay,
                            'step_size': step_size,
                            'gamma': gamma
                        }

# Print out the best parameters found
print(f"Best hyperparameters found:\n{best_params}")
