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
# Create my own class with the layers and the forward step based on PyTorch
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
        self.auROC = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    #Here is new code, now we define our proper training, validation and test step
    #For all of them return the loss result by the common step!
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'train_loss':loss, 'train_accuracy':accuracy, 'train_f1_score':f1_score, 'train_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 100 == 0:
            samples = x[:8]
            grid = torchvision.utils.make_grid(samples.view(-1, 1, 28, 28))
            #self.logger.experiment.add_image("mnist_images", grid, self.global_step)

        return {'loss': loss, "scores": scores, "y": y}
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'val_loss':loss, 'val_accuracy':accuracy, 'val_f1_score':f1_score, 'val_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'test_loss':loss, 'test_accuracy':accuracy, 'test_f1_score':f1_score, 'test_auROC':auROC}, 
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
        return optim.Adam(self.parameters(), lr=self.learning_rate)

