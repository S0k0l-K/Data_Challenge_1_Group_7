# Data-Challenge-1

This repository is for the TU/e course JBG040 Data Challenge 1. It contains the necessary code to train a neural network on the provided image dataset. Below you will find instructions on how to set up your environment, run the data download script, train your neural network, and submit your code.

## Code Structure

The repository includes several Python files, organized as follows:

- `ImageDataset.py`: Downloads the image data and saves it to `/data/`.
- `mobilenet.py`: Contains the definition of the neural network, loss functions, and optimizers.
- `batch_sampler.py` & `image_dataset.py`: Define custom PyTorch `Dataset` and `Sampler` for loading and batching the data.
- Additional files for utilities, training, and testing.

## Initial Setup

1. Clone the repository to your local machine.
2. Navigate to the repository folder using a terminal or command prompt.
3. Set up a virtual environment using Python 3.8+ and activate it.
4. Install the required dependencies with `pip install -r requirements.txt`.

## Downloading Data

Before training your model, you need to download the dataset:

1. Run `python ImageDataset.py` to download and save the training and test data into the `/data/` directory.
2. You only need to perform this step once.

## Training and Evaluation

To train and evaluate the neural network:

1. Ensure the dataset is downloaded as described in the previous section.
2. Run `mobilenet.py` with your desired parameters. For example:

```bash
python mobilenet.py --nb_epochs 10 --batch_size 64 --balanced_batches
