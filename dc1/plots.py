import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from mobilenet import CustomDataset
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

class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomMobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False)
        in_features = self.mobilenet.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    pass

# Load saved model weights
model = CustomMobileNet(num_classes=6)
model.load_state_dict(torch.load("MobilNetFinal.pth"))
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

scores = np.array(scores)
true_labels_binarized = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5])

# Plot confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')
plt.show()

for i in range(6):
    precision, recall, _ = precision_recall_curve(true_labels_binarized[:, i], scores[:, i])
    plt.plot(recall, precision, lw=2, label='class {}'.format(i))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision-Recall curve")
plt.show()
