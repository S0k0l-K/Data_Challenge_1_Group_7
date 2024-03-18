# Custom imports
from torch.utils.data import WeightedRandomSampler, DataLoader

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset

from dc1.net import Net
from dc1.train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List

from dc1.alexnet import AlexNet
import numpy as np
from torchvision import transforms

from sklearn.metrics import classification_report

def evaluate_model(model, test_dataset, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_dataset:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    target_names =['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Compute general accuracy, F1 score, precision, and recall
    accuracy = (all_preds == all_labels).mean()
    f1 = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)['macro avg']['f1-score']
    precision = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)['macro avg']['precision']
    recall = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)['macro avg']['recall']

    print(f"General Accuracy: {accuracy}")
    print(f"General F1 Score: {f1}")
    print(f"General Precision: {precision}")
    print(f"General Recall: {recall}")

    return accuracy, f1, precision, recall




def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    #train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
    #test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy arrays to PIL images first if necessary
        transforms.Grayscale(num_output_channels=1),  # Ensure images are greyscale
        # Apply other transformations here, such as resizing or normalization
        transforms.ToTensor(),  # Convert PIL images to tensors
    ])

    train_dataset = ImageDataset(Path("../data/resized_X_train1.npy"), Path("../data/Y_train.npy"), transform=transform)
    test_dataset = ImageDataset(Path("../data/resized_X_test1.npy"), Path("../data/Y_test.npy"), transform=transform)

    #Data Stratification
    #train_labels = train_d.targets  # Now directly accessing the labels
    #class_sample_counts = [2521, 2318, 2964, 6103, 1633, 1302]
    #total_samples = sum(class_sample_counts)
    #class_weights = [total_samples / count for count in class_sample_counts]
    # List of weights
    #sample_weights = [class_weights[label] for label in train_labels]
    # Creating a sampler
    #sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # DataLoader for training and testing, now using the sampler for balanced classes
    #train_sampler = DataLoader(train_d, batch_size=args.batch_size, sampler=sampler)
    #test_sampler= DataLoader(test_d, batch_size=args.batch_size, shuffle=False)


    # Load the Neural Net. NOTE: set number of distinct labels here
    #model = Net(n_classes=6)
    model= AlexNet(num_classes=6)
    # Initialize optimizer(s) and loss function(s)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        #summary(model, (1, 128, 128), device=device)
        summary(model, (1, 256, 256), device=device)
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        #summary(model, (1, 128, 128), device=device)
        summary(model, (1, 256, 256), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)





    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    
    for e in range(n_epochs):
        if activeloop:



            # Training:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            losses = test_model(model, test_sampler, loss_function, device)

            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    #Evaluation

    accuracy, f1, precision, recall = evaluate_model(model, test_sampler, device)

    # Saving the model

    model_file_path = f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.pth"
    torch.save(model.state_dict(), model_file_path)
    print(f"Trained model saved at: {model_file_path}")

    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")



    # Evaluation after training
    #evaluate_model(model, test_dataset, loss_function, device)


    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)
