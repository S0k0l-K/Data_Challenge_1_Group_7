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

python mobilenet.py --nb_epochs 10 --batch_size 64 --balanced_batches

## Additional Information for Plotting Functionality

To leverage the plotting functionality provided in `plots.py`, follow these steps:

1. Ensure `matplotlib` and `seaborn` are installed in your environment. These can be installed via pip if not already included in `requirements.txt`:
    ```bash
    pip install matplotlib seaborn
    ```
2. After training your model using `main.py`, `plots.py` can be used to generate visualizations such as loss curves and accuracy plots. To use `plots.py`, make sure it imports the required data from your training session, such as loss values and accuracy metrics. You might need to modify `plots.py` to correctly locate and load this data.
3. Run `plots.py` with the Python command, specifying any required arguments. If arguments are implemented, use something like:
    ```bash
    python plots.py --loss_path=/path/to/loss_data.npy --accuracy_path=/path/to/accuracy_data.npy
    ```
    Replace `/path/to/loss_data.npy` and `/path/to/accuracy_data.npy` with the actual paths to your saved loss and accuracy data files.
4. `plots.py` will generate the plots and save them in a specified directory. Ensure this directory is correctly set within `plots.py` to avoid file not found errors.

This setup assumes you have implemented or will implement a script named `plots.py` for generating visualizations. Modify the instructions based on the actual implementation details of your plotting script.

## Plotting for baseline model

1. First run the `Net.py` file and the `train_test.py` file.
2. Second run the `main.py` file and be sure that the `Netbaseline.pth` is there in your files.
3. Run `plots baseline.py` to get the plots from the baseline model.

If you run into any errors from the import part be sure to pip install that certain import
