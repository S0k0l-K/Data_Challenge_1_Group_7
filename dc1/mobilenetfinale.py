import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import torch.optim as optim

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
        image = image.astype(np.uint8)
        image = image.reshape(128, 128, 1)  # Ensure image is in the format (H, W, C)
        if self.transform:
            image = self.transform(image)
        return image, label

# Enhanced data augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5113557577133179], std=[0.24621723592281342]),
])

# Load datasets
data_path = Path(__file__).parent.parent
train_x_path = data_path / "data/X_train.npy"
train_y_path = data_path / "data/Y_train.npy"
test_x_path = data_path / "data/X_test.npy"
test_y_path = data_path / "data/Y_test.npy"

x_train, y_train = np.load(train_x_path), np.load(train_y_path)
x_test, y_test = np.load(test_x_path), np.load(test_y_path)

train_dataset = CustomDataset(x_train, y_train, transform=transform)
test_dataset = CustomDataset(x_test, y_test, transform=transform)

# Initialize DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Adjusted FocalLoss with class weights
class_counts = [np.sum(y_train == i) for i in range(6)]
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32).to(device)
class_weights = class_weights / class_weights.sum()  # Normalize weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)
        if self.weight is not None:
            alpha_factor = self.weight.gather(0, targets.data.view(-1))
        else:
            alpha_factor = torch.tensor(self.alpha, device=inputs.device)
        alpha_factor = alpha_factor.view(-1, 1)
        F_loss = alpha_factor * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

criterion = FocalLoss(weight=class_weights, alpha=0.25, gamma=2.0, reduction='mean').to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

num_epochs = 10

best_val_accuracy = 0.0
model_save_path = 'MobileNet_best_val_accuracy.pth'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()  # Convert labels to torch.long
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total * 100

    # Validation step
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            test_preds.extend(predicted.view(-1).cpu().numpy())
            test_targets.extend(labels.view(-1).cpu().numpy())

    test_loss = val_running_loss / len(test_loader)
    test_accuracy = val_correct / val_total * 100

    # Checkpointing
    if test_accuracy > best_val_accuracy:
        best_val_accuracy = test_accuracy
        torch.save(model.state_dict(), model_save_path)

    scheduler.step(test_loss)  # Adjust the learning rate based on the validation loss

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}')

# Final metrics and classification report after training
test_f1 = f1_score(test_targets, test_preds, average='weighted')
class_names = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule', 'Pneumothorax']
print("Final Test Classification Report:")
print(classification_report(test_targets, test_preds, target_names=class_names))
final_model_save_path = 'MobileNet_final_epoch.pth'
torch.save(model.state_dict(), final_model_save_path)