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
from model import TitanicModel
from dataset import TitanicDataModule
import torch.multiprocessing
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="mnist_config", node=Configuration)
@hydra.main(config_path="config", config_name="config")
def main(config : Configuration):

    logger_wand = WandbLogger(name='wand_logs', project='Titanic')
    myDataModule = TitanicDataModule(data_dir=config.dataset.DATA_DIR, batch_size=config.train.BATCH_SIZE, num_workers=config.train.NUM_WORKERS)
    model = TitanicModel(input_size=config.model.INPUT_SIZE, num_classes=config.model.NUM_CLASSES, learning_rate=config.train.LEARNING_RATE).to(config.train.ACCELERATOR)
    trainer = pl.Trainer(
        logger=logger_wand,
        accelerator=config.train.ACCELERATOR,
        min_epochs=1,
        max_epochs=config.train.NUM_EPOCHS,
        precision=config.train.PRECISION,
        log_every_n_steps=config.train.LOG_EVERY_STEPS
        )
    trainer.fit(model, myDataModule)
    #trainer.validate(model, myDataModule)
    trainer.test(model, myDataModule)
    print("Success! ")

if __name__ == "__main__":
    main()