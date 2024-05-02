import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl

# Create my own class with the layers and the forward step based on PyTorch
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    #Here is new code, now we define our proper training, validation and test step
    #For all of them return the loss result by the common step!
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    '''
    This is our common step used by the traning, validation and test step
    '''
    def _common_step(self, batch, batch_idx):
        #Get the samples and labels from the batch and apply the reshape 
        x, y = batch 
        x = x.reshape(x.size(0), -1)
        #Compute the forward step and get the scores
        scores = self.forward(x)
        #Apply the loss function
        loss = self.loss_fn(scores, y)
        #Return the loss, the scores, and the labels
        return loss, scores, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch 
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3
# Load the data from datasets MNIST
# This download in case we don't have downloaded before.
# Apply transformation to Tensor
entire_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
#Split the dataset in train and validation dataset
train_ds, val_ds = random_split(entire_dataset, [50000, 10000])

#Get the test dataset and also apply transformation to Tensor
test_ds = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)

#Set each dataset onto a DataLoader
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False, num_workers=12)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=12)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = pl.Trainer(accelerator='auto', min_epochs=1, max_epochs=num_epochs, precision=16)
trainer.fit(model, train_loader, val_loader)
'''
Check the accuracy of our model
'''
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    #Set the model in eval mode
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for batch_idx, (x, y) in enumerate(tqdm(loader)):

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape
            # (64, 720)
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# Check accuracy on training & test to see how good our model
model.to(device)
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")