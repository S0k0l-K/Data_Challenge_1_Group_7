import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from ResNetAUG import CustomResNet



def prepare_dataset(x, y):

    if x.ndim == 5:
        x = x.squeeze(2)
    elif x.ndim == 3:
        x = np.expand_dims(x, axis=1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds,
                                target_names=['Pneumothorax', 'Nodule', 'Infiltration', 'Effusion', 'Atelectasis',
                                              'No Finding']))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pneumothorax', 'Nodule', 'Infiltration', 'Effusion', 'Atelectasis', 'No Finding'],
                yticklabels=['Pneumothorax', 'Nodule', 'Infiltration', 'Effusion', 'Atelectasis', 'No Finding'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    data_path = Path(__file__).parent
    test_x_path = data_path / "X_test.npy"
    test_y_path = data_path / "Y_test.npy"

    x_test, y_test = np.load(test_x_path), np.load(test_y_path)
    test_dataset = prepare_dataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=6).to(device)
    model_path = data_path / 'custom_resnet_model_augmented.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
