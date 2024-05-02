
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import pandas as pd
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import pytorch_lightning as pl
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip

class TitaticDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        features = torch.tensor([
        sample.at['Pclass'],
        sample.at['Sex'],
        sample.at['Age'],
        sample.at['SibSp'],
        sample.at['Parch'],
        sample.at['Fare'],
        sample.at['Embarked']
    ], dtype=torch.float32)
        label = torch.tensor(sample['Survived'], dtype=torch.float32)
        return features, label

class TitanicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: str):
        self.train = pd.read_csv(self.data_dir + "train_ds.csv")
        self.test = pd.read_csv(self.data_dir +'test_ds.csv')
    
    def train_dataloader(self):
        custom_dataset = TitaticDataset(self.train)
        return DataLoader(
            custom_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    def test_dataloader(self):

        custom_dataset = TitaticDataset(self.test)
        return DataLoader(
            custom_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )