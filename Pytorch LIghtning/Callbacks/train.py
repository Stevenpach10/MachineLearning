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
from model import NN
from dataset import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
import torch.multiprocessing


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    myDataModule = MnistDataModule(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    # Initialize network
    model = NN(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE).to(config.ACCELERATOR)

    trainer = pl.Trainer(accelerator=config.ACCELERATOR,
                         min_epochs=1,
                         max_epochs=config.NUM_EPOCHS,
                         precision=config.PRECISION,
                         callbacks=[MyPrintingCallback(), EarlyStopping(monitor='val_loss')])
    trainer.fit(model, myDataModule)
    trainer.validate(model, myDataModule)
    trainer.test(model, myDataModule)