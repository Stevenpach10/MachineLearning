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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import wandb
from torchvision.utils import make_grid


class DeconvolutionalBlock(pl.LightningModule):
    def __init__(self, in_channels_block, out_channels_block, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels_block = in_channels_block
        self.out_channels_block = out_channels_block
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels_block, out_channels=self.out_channels_block, kernel_size=kernel_size, stride=stride, padding=padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.out_channels_block, out_channels=self.out_channels_block, kernel_size=kernel_size, stride=stride, padding=padding)
    
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        

    def forward(self, x, skip_connection=None):
        x = self.upsample(x)  # Upsample
        if skip_connection is not None:
            x = torch.cat((x, skip_connection), dim=1)
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        
        
        return x
    
class ConvolutionalBlock(pl.LightningModule):
    def __init__(self, in_channels_block, out_channels_block, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.in_channels_block = in_channels_block
        self.out_channels_block = out_channels_block

        self.conv1 = nn.Conv2d(in_channels=in_channels_block, out_channels=self.out_channels_block, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels_block, out_channels=self.out_channels_block, kernel_size=kernel_size, stride=stride, padding=padding)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        self.skip_connection = x
        x = self.pool(x)
        return x
    
class Autoencoder(pl.LightningModule):
    def __init__(self, input_channels, num_classes, learning_rate):
        super(Autoencoder, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.conv_block = ConvolutionalBlock(in_channels_block=input_channels, out_channels_block=16)
        self.conv_block2 = ConvolutionalBlock(in_channels_block=self.conv_block.out_channels_block, out_channels_block=32)
        self.conv_block3 = ConvolutionalBlock(in_channels_block=self.conv_block2.out_channels_block, out_channels_block=64)
        self.conv_block4 = ConvolutionalBlock(in_channels_block=self.conv_block3.out_channels_block, out_channels_block=128)

        # Input channel by 2 due to the skip connections
        self.deconv_block4 = DeconvolutionalBlock(in_channels_block=self.conv_block4.out_channels_block*2, out_channels_block=64)
        self.deconv_block3 = DeconvolutionalBlock(in_channels_block=self.deconv_block4.out_channels_block*2, out_channels_block=32)
        self.deconv_block2 = DeconvolutionalBlock(in_channels_block=self.deconv_block3.out_channels_block*2, out_channels_block=16)
        self.deconv_block = DeconvolutionalBlock(in_channels_block=self.deconv_block2.out_channels_block*2, out_channels_block=input_channels)
        self.loss_fn = nn.MSELoss()  # Usando Mean Squared Error (MSE) para la reconstrucción
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.testing_step_outputs = []

    # (batch_size, 3, 224,224)
    def forward(self, x):
        x = self.conv_block(x)
        saved_skip_connection = self.conv_block.skip_connection
        # (batch_size, 16, 112, 112)
        x = self.conv_block2(x)
        saved_skip_connection2 = self.conv_block2.skip_connection
        # (batch_size, 32, 56, 56)
        x = self.conv_block3(x)
        saved_skip_connection3 = self.conv_block3.skip_connection

        # (batch_size, 64, 28, 28)
        x = self.conv_block4(x)
        saved_skip_connection4 = self.conv_block4.skip_connection
        
        # (batch_size, 768, 14, 14)
        x = self.deconv_block4(x, saved_skip_connection4)
        x = self.deconv_block3(x, saved_skip_connection3)
        x = self.deconv_block2(x, saved_skip_connection2)
        x = self.deconv_block(x, saved_skip_connection)
        return x
    
    def test_step(self, batch, batch_idx):
        x, _ = batch 
        loss = self._common_step(batch, batch_idx)
        self.testing_step_outputs.append(loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch 
        loss = self._common_step(batch, batch_idx)
        self.validation_step_outputs.append(loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch 
        loss = self._common_step(batch, batch_idx)
        self.training_step_outputs.append(loss)

        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        print(f'Epoch {self.current_epoch}, Training Loss: {avg_loss.item()}', end=None)
        self.log('train_loss', avg_loss.item())
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        print(f'Epoch {self.current_epoch}, Validation Loss: {avg_loss.item()}', end=None)
        self.log('validation_loss', avg_loss.item())
        self.validation_step_outputs.clear()

        
        self.save_reconstructed_images()
    
    def on_testing_epoch_end(self):
        avg_loss = torch.stack(self.testing_step_outputs).mean()
        print(f'Epoch {self.current_epoch}, Testing Loss: {avg_loss.item()}', end=None)
        self.log('test_loss', avg_loss.item())
        self.testing_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }

    def save_reconstructed_images(self):
        # Assuming you have a validation data loader available
        val_loader = self.trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        x, _ = batch
        x = x.to(self.device)
        with torch.no_grad():
            x_hat = self(x)

        # Convert images to a grid
        original_grid = make_grid(x)
        reconstructed_grid = make_grid(x_hat)

        # Log images to W&B
        if self.logger and isinstance(self.logger.experiment, wandb.wandb_run.Run):
            self.logger.experiment.log({
                f'original_images': [wandb.Image(original_grid)],
                f'reconstructed_images': [wandb.Image(reconstructed_grid)]
            }) 

    def _common_step(self, batch, batch_idx):
        x, y = batch 
        reconstructed_x = self.forward(x)
        loss = self.loss_fn(reconstructed_x, x)
        return loss