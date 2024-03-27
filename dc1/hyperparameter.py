# hyperparameter.py
from mobilenetoptimized_loss_function import CustomDataset, FocalLoss, train_model
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

# Define device: use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
# Make sure to define transform or import it if it's defined in mobilenetoptimized_loss_function.py
data_path = Path('data')  # Adjust as needed
train_x_path = data_path / "X_train.npy"
train_y_path = data_path / "Y_train.npy"
test_x_path = data_path / "X_test.npy"
test_y_path = data_path / "Y_test.npy"

x_train, y_train = np.load(train_x_path), np.load(train_y_path)
x_test, y_test = np.load(test_x_path), np.load(test_y_path)

# Initialize datasets
train_dataset = CustomDataset(x_train, y_train, transform=transform)
test_dataset = CustomDataset(x_test, y_test, transform=transform)

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
                    model = models.mobilenet_v2(pretrained=False)
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
