from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb

from torch import nn, optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from config.config import Configuration
from model import CCN
from dataset import HandsDataModule
import torch.multiprocessing
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="configs", node=Configuration)
@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config : Configuration):
    handsDataModule = HandsDataModule(data_dir=config.dataset.DATA_DIR,
                                      batch_size=config.train.BATCH_SIZE, 
                                      num_workers=config.train.NUM_WORKERS,
                                      is_reduce_dataset=config.train.REDUCE_DATASET,
                                      dataset_size=config.train.DATASET_SIZE)
    
    model = CCN(in_channels=config.model.IN_CHANNELS , num_classes=config.model.NUM_CLASSES, learning_rate=config.train.LEARNING_RATE, batch_size=config.train.BATCH_SIZE)
    trainer = pl.Trainer(
        accelerator=config.train.ACCELERATOR,
        min_epochs=1,
        max_epochs=config.train.NUM_EPOCHS,
        precision=config.train.PRECISION,
    )
    trainer.fit(model, handsDataModule)
    trainer.test(model, handsDataModule)
    trainer.validate(model, handsDataModule)
    
if __name__ == "__main__":
    main()