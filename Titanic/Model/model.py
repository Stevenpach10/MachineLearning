from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
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
import pytorch_lightning as pl
class TitanicModel(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 150)
        self.fc3 = nn.Linear(150, num_classes)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.loss_function = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="binary", num_classes=num_classes)
        self.auROC = torchmetrics.AUROC(task="binary", num_classes=num_classes)
        self.training_step_outputs = []
    
    #Here is new code, now we define our proper training, validation and test step
    #For all of them return the loss result by the common step!
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.training_step_outputs.append({'loss': loss, "scores": scores, "y": y})
        return {'loss': loss, "scores": scores, "y": y}
    
    def on_train_epoch_start(self):
        self.training_step_outputs = []
    
    def on_train_epoch_end(self):
        #Takes the scores and labels from all the outputs of the training step
        scores = torch.cat([x["scores"] for x in self.training_step_outputs])
        y = torch.cat([x["y"] for x in self.training_step_outputs])
        loss = torch.tensor([x["loss"] for x in self.training_step_outputs])
        self.log_dict({
            "loss" : loss.mean(),
            "train_acc": self.accuracy(scores, y),
            "train_f1" : self.f1(scores, y),
            "auROC" : self.auROC(scores, y)
        }, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'val_loss':loss, 'val_accuracy':accuracy, 'val_f1_score':f1_score, 'val_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'test_loss':loss, 'test_accuracy':accuracy, 'test_f1_score':f1_score, 'test_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def _common_step(self, batch):
        x, y = batch
        scores = torch.sigmoid(self.forward(x))
        scores = torch.squeeze(scores, dim=1)
        loss = self.loss_function(scores, y)
        return loss, scores, y
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)