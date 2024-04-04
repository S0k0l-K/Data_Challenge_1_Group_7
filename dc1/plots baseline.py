import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
# from mobilenet import CustomDataset
from dc1.net import Net
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
import plotext  # type: ignore
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.optim import Adam
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            torch.nn.Dropout(p=0.5, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            torch.nn.Dropout(p=0.25, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.125, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.Linear(256, num_classes)
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
# Load saved model weights
model = Net(num_classes=6)
model.load_state_dict(torch.load("Netbaseline.pth"))
model.eval()

data_path = Path(__file__).parent.parent
train_x_path = data_path / "data/X_train.npy"
train_y_path = data_path / "data/Y_train.npy"
test_x_path = data_path / "data/X_test.npy"
test_y_path = data_path / "data/Y_test.npy"

x_train, y_train = np.load(train_x_path), np.load(train_y_path)
x_test, y_test = np.load(test_x_path), np.load(test_y_path)

train_dataset = ImageDataset(train_x_path, train_y_path)
test_dataset = ImageDataset(test_x_path, test_y_path)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate predictions and true labels
true_labels = []
predictions = []
scores = []
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        output = model(images)
        _, preds = torch.max(output, 1)
        predictions.extend(preds.view(-1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        scores.extend(torch.softmax(output, dim=1).cpu().numpy())

# Convert scores and true_labels for multi-class precision-recall curve
scores = np.array(scores)
true_labels_binarized = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5])

# Generate classification report
print("Classification Report:")
print(classification_report(true_labels, predictions))

# Plot confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')
plt.show()

# Plot precision-recall curve for each class
for i in range(6): # Assuming 6 classes
    precision, recall, _ = precision_recall_curve(true_labels_binarized[:, i], scores[:, i])
    plt.plot(recall, precision, lw=2, label='class {}'.format(i))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision-Recall curve")
plt.show()