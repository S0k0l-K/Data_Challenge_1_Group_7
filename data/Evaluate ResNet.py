import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from ResNet import CustomResNet
from ResNet import load_data_and_create_dataset


def evaluate_model(model, test_loader):
    model.eval() 
    true_labels = []
    pred_labels = []

    with torch.no_grad():  
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def main():
    model_path = Path(__file__).parent / "resnet_model.pth"
    model = CustomResNet(num_classes=6)
    model.load_state_dict(torch.load(model_path))

    x_test_path = Path(__file__).parent.parent / "data" / "X_test.npy"
    y_test_path = Path(__file__).parent.parent / "data" / "Y_test.npy"
    test_dataset = load_data_and_create_dataset(x_test_path, y_test_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
