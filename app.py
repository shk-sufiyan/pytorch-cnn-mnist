# Import pytorch dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

## setting variables
PROCESS_TYPE = "mps"
NUMBER_OF_EPOCH = 10
LEARNING_RATE = 1e-3
MODEL_NAME = "MNIST_CNN.pt"

## fetch training data from MNIST
train_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train_data, 32)

## Model class for Image classification
class ImageClassificartion(nn.Module):

    ## init function and construct the model
    def __init__(self):
        super().__init__()

        self.model = self.create_cnn_model()
        pass

    ## forward function
    def forward(self, instance):
        return self.model(instance)

    ## contruct cnn model
    def create_cnn_model(self):
        model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),

            ## now flat the layer
            nn.Flatten(),

            ## output layer as per the requured data
            ## image pixel - 28 x 28
            ## image classes - 0 - 9 = 10
            nn.Linear(64*(28-6)*(28-6), 10)
        )
        return model



## Insace of the CNN
clf = ImageClassificartion().to(PROCESS_TYPE)

## Optimizer
opt = Adam(clf.parameters(), lr=LEARNING_RATE)

## Loss function
loss_fn = nn.CrossEntropyLoss()


## Training function
def model_training():
    for epoch in range(NUMBER_OF_EPOCH):
        for batch in dataset:

            X, Y = batch
            X, Y = X.to(PROCESS_TYPE), Y.to(PROCESS_TYPE)
            Yhat = clf(X) ## generate a prediction

            ## calculate loss
            loss = loss_fn(Yhat, Y)

            ## Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        ## print the training output
        print(f"Current Epoch: {epoch} loss is {loss.item()}")

    ## save the trained model
    with open(MODEL_NAME, 'wb') as f:
        save(clf.state_dict(), f)


## model testing or usage
def use_model(image_path):
    with open(MODEL_NAME, 'rb') as f:
        clf.load_state_dict(load(f))

    ## load image
    image = Image.open(image_path)
    image_tensor = ToTensor()(image).unsqueeze(0).to(PROCESS_TYPE)

    ## print the prediction
    print(torch.argmax(clf(image_tensor)))




## main function call
if __name__ == "__main__":

    # model_training() ## call this function for training the model

    ## test the model
    image_path = "images/digit-2.jpg"
    use_model(image_path)


