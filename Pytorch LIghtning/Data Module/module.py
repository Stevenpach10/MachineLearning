from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric 

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()
        
    def compute(self):
        return self.correct.float() / self.total.float()


# Create my own class with the layers and the forward step based on PyTorch
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.myAccuracy = Accuracy()
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
        self.auROC = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    #Here is new code, now we define our proper training, validation and test step
    #For all of them return the loss result by the common step!
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        myAccuracy = self.myAccuracy(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'train_loss':loss, 'train_accuracy':accuracy, 'train_my_accuracy' : myAccuracy, 'train_f1_score':f1_score, 'train_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, "scores": scores, "y": y}
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        myAccuracy = self.myAccuracy(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'val_loss':loss, 'val_accuracy':accuracy, 'val_my_accuracy' : myAccuracy, 'val_f1_score':f1_score, 'val_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        myAccuracy = self.myAccuracy(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'test_loss':loss, 'test_accuracy':accuracy, 'test_my_accuracy' : myAccuracy, 'test_f1_score':f1_score, 'test_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
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


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    #Here we can donwload the data
    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)
    #Here we can load the data
    def setup(self, stage: str):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000]) 

        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

myDataModule = MnistDataModule(data_dir='dataset/', batch_size=batch_size, num_workers=4)
# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

trainer = pl.Trainer(accelerator='auto', min_epochs=1, max_epochs=num_epochs, precision=16)
trainer.fit(model, myDataModule)
trainer.validate(model, myDataModule)
trainer.test(model, myDataModule)