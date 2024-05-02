from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric 

class CCN(pl.LightningModule):
    def __init__(self, in_channels, num_classes, learning_rate, batch_size):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(48*15*15, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3123 = nn.Linear(16, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.auROC = torchmetrics.AUROC(task='binary')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss':loss, 'train_accuracy':accuracy, 'train_f1_score':f1_score}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, "scores": scores, "y": y}
    
    def trainin_epoch_end(self, outputs):
        #Takes the scores and labels from all the outputs of the training step
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        self.log_dict({
            "train_acc": self.accuracy(scores, y),
            "train_f1" : self.f1_score(scores, y),
            "auROC" : self.auROC(scores, y)
        }, on_step=False, on_epoch=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        #auROC = self.auROC(scores, y)
        self.log_dict({'val_loss':loss, 'val_accuracy':accuracy, 'val_f1_score':f1_score}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, "scores": scores, "y": y}

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        #auROC = self.auROC(scores, y)
        self.log_dict({'test_loss':loss, 'test_accuracy':accuracy, 'test_f1_score':f1_score}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, "scores": scores, "y": y}
        
    
    def _common_step(self, batch, batch_idx):
        #Get the samples and labels from the batch and apply the reshape 
        x, y = batch
        #Compute the forward step and get the scores
        scores = self.forward(x)
        scores = scores.view(scores.shape[0])
        #Apply the loss function
        loss = self.loss_fn(scores, y.float())
        #Return the loss, the scores, and the labels
        return loss, scores, y
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    
