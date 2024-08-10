# PyTorch CNN for MNIST Dataset

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using PyTorch to classify images from the MNIST dataset. The MNIST dataset consists of handwritten digits (0-9) and is commonly used for training image classification models.

## Project Structure

1. **Model Definition**: Defines a CNN model for image classification.
2. **Training**: Trains the model on the MNIST dataset.
3. **Testing**: Uses the trained model to make predictions on new images.

## Prerequisites

Before running the code, ensure you have the following Python packages installed:

- `torch` (PyTorch)
- `PIL` (Pillow)
- `torchvision`

## Setup and installation

First you have to clone the repo, and then create a python3 environement and install the requirements

```bash
git clone git@github.com:shk-sufiyan/pytorch-cnn-mnist.git
cd pytorch-cnn-mnist
python3.10 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```

## Code Overview

### Import Dependencies

```python
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

### Settings

- PROCESS_TYPE: Determines the computation device (e.g., "mps" for Apple Silicon, "cuda" for GPUs, "cpu" for CPU).
- NUMBER_OF_EPOCH: Number of epochs for training.
- LEARNING_RATE: Learning rate for the optimizer.
- MODEL_NAME: Name of the file to save the trained model.

### Data Loading

```python
train_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train_data, 32)
```

### Model Definition

The `ImageClassificartion` class defines the CNN model:

```python
class ImageClassificartion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.create_cnn_model()

    def forward(self, instance):
        return self.model(instance)

    def create_cnn_model(self):
        model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )
        return model
```

### Training

The `model_training` function trains the model and saves it:

```python
def model_training():
    for epoch in range(NUMBER_OF_EPOCH):
        for batch in dataset:
            X, Y = batch
            X, Y = X.to(PROCESS_TYPE), Y.to(PROCESS_TYPE)
            Yhat = clf(X)
            loss = loss_fn(Yhat, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Current Epoch: {epoch} loss is {loss.item()}")
    with open(MODEL_NAME, 'wb') as f:
        save(clf.state_dict(), f)
```

### Model Testing

The `use_model` function loads a saved model and makes predictions:

```python
def use_model(image_path):
    with open(MODEL_NAME, 'rb') as f:
        clf.load_state_dict(load(f))
    image = Image.open(image_path)
    image_tensor = ToTensor()(image).unsqueeze(0).to(PROCESS_TYPE)
    print(torch.argmax(clf(image_tensor)))
```

### Main Function

The entry point of the script:

```python
if __name__ == "__main__":
    # model_training() # Uncomment to train the model
    image_path = "images/digit-2.jpg"
    use_model(image_path)
```

## Usage

1. Train the Model: Uncomment the `model_training()` call in the main function and run the script to train the model.
2. Test the Model: After training, or if you have a pre-trained model, specify the path to an image file and run the script to get predictions.

## Notes

- Ensure the image used for testing is in the correct format and size (28x28 pixels).
- The model and its parameters are saved in `MNIST_CNN.pt`.