
from torch.utils.data import Dataset, DataLoader
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
import plotext  # type: ignore
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.optim import Adam


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # Convert image to uint8
        image = image.astype(np.uint8)
        # Ensure image is in the format (H, W, C)
        image = image.reshape(128, 128, 1)
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5113557577133179], std=[0.24621723592281342]),
    transforms.RandomAffine(degrees=15, scale=(0.9, 1.1))
])

data_path = Path(__file__).parent.parent
train_x_path = data_path / "data/X_train.npy"
train_y_path = data_path / "data/Y_train.npy"
test_x_path = data_path / "data/X_test.npy"
test_y_path = data_path / "data/Y_test.npy"

x_train, y_train = np.load(train_x_path), np.load(train_y_path)
x_test, y_test = np.load(test_x_path), np.load(test_y_path)

train_dataset = ImageDataset(train_x_path, train_y_path)
test_dataset = ImageDataset(test_x_path, test_y_path)


class_counts = [np.sum(y_train == i) for i in range(6)]
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
class_weights = class_weights / class_weights.sum()  # Normalize weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)
        alpha_factor = self.weight[targets] if self.weight is not None else self.alpha
        F_loss = alpha_factor * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_batch_sampler = BatchSampler(batch_size=64, dataset=train_dataset, balanced=True)
test_batch_sampler = BatchSampler(batch_size=64, dataset=test_dataset, balanced=True)


model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = FocalLoss(alpha=1, gamma=2).to(device)
optimizer = Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_preds = []
    train_targets = []

    for images, labels in train_batch_sampler:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / len(train_batch_sampler)
    train_accuracy = 100.0 * correct / total
    train_precision = precision_score(train_targets, train_preds, average=None)
    train_recall = recall_score(train_targets, train_preds, average=None)
    class_names = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding','Nodule','Pneumothorax']
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    test_preds = []
    test_targets = []

    class_index_to_name = {i: class_names[i] for i in range(len(class_names))}

    with torch.no_grad():
        for images, labels in test_batch_sampler:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            predicted_classes = [class_index_to_name[p.item()] for p in predicted]
            true_classes = [class_index_to_name[l.item()] for l in labels]
    test_loss /= len(test_batch_sampler)
    test_accuracy = 100.0 * correct / total
    test_precision = precision_score(test_targets, test_preds, average=None)
    test_recall = recall_score(test_targets, test_preds, average=None)

    torch.save(model.state_dict(), 'MobilNetFinal.pth')
    print("Train Classification Report:")
    print(classification_report(train_targets, train_preds,
                                target_names=[class_index_to_name[i] for i in range(len(class_names))]))
    print("Test Classification Report:")
    print(classification_report(test_targets, test_preds,
                                target_names=[class_index_to_name[i] for i in range(len(class_names))]))
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

torch.save(model.state_dict(), 'MobilNetFinal1.pth')