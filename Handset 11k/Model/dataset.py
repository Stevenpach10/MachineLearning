
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
import glob
import shutil
import csv
import random

from shutil import copyfile
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets.folder import default_loader

from os import walk

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = glob.glob(os.path.join(self.root, '*.jpg'))
    
    def __getitem__(self, index):
        path = self.samples[index]
        sample = read_image(path).float() 
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            filename = os.path.basename(path)
            target = self.target_transform(filename)

        return sample, target
    def __len__(self):
        return len(self.samples)

class HandsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, is_reduce_dataset, dataset_size, image_size=(255, 255)):
        super().__init__()
        self.is_reduce_dataset = is_reduce_dataset
        self.dataset_size = dataset_size
        if not self.is_reduce_dataset:
            self.data_dir = data_dir
        else:
            self.data_dir = data_dir
            self.data_dir_reduce = data_dir + "/" + "reduce"
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.image_size = image_size

    def prepare_data(self) -> None:
        # Create train and test directories
        self.train_path = os.path.join(self.data_dir, 'train')
        self.test_path = os.path.join(self.data_dir, 'test')

        if self.is_reduce_dataset:
            #If reduce dataset flag is True
            #We're going to create a separate folder to keep the training and testing set
            self.train_path = os.path.join(self.data_dir_reduce, 'train')
            self.test_path = os.path.join(self.data_dir_reduce, 'test')
            training_folder = glob.glob(os.path.join(self.train_path, '*.jpg'))
            testing_folder = glob.glob(os.path.join(self.test_path, '*.jpg'))
            #If size of training set of sample change, delete it to create it again!
            
            if(len(training_folder) != int((self.dataset_size+1)*.80)):
                shutil.rmtree(self.train_path)
                shutil.rmtree(self.test_path)
            else:
                print("Total training dataset : %3d, testing set : %2d" % (len(training_folder), len(testing_folder))) 
        
        #Load the labels:
        # Read the labels from the CSV file
        self.image_label_mapping = {}
        with open(self.data_dir+"/HandInfo.csv", 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row
            for row in csvreader:
                _,age,gender,_,_,_,_,image_name,_ = row  # Adjust this based on your CSV format
                sex = 0
                if gender == "male":
                    sex = 1
                self.image_label_mapping[image_name] = sex  # Convert label to an integer if needed

        #If training set exists keep it.
        if not os.path.exists(self.train_path):
            print("Creating folder for training and testing set")
            os.makedirs(self.train_path, exist_ok=True)
            os.makedirs(self.test_path, exist_ok=True)
            image_files = glob.glob(os.path.join(self.data_dir, '*.jpg'))

            random.shuffle(image_files)
            size = len(image_files)
            if self.is_reduce_dataset:
                size = self.dataset_size
            size_training_set = int((size+1)*.80)
               
            training_set = image_files[:size_training_set]
            testing_set = image_files[size_training_set:size]

            #Move the files into de training set
            for image_file in training_set:
                target_path_training = os.path.join(self.train_path, os.path.basename(image_file))
                if not os.path.exists(target_path_training):
                    copyfile(image_file, target_path_training)
            
            #Move the files into de testing set
            for image_file in testing_set:
                target_path_test = os.path.join(self.test_path, os.path.basename(image_file))
                if not os.path.exists(target_path_test):
                    copyfile(image_file, target_path_test)
            
            print("Total training dataset : %3d, testing set : %2d" % (len(training_set), len(testing_set))) 
         

    #Here we can load the data
    def setup(self, stage: str):
        if self.is_reduce_dataset:
            directory = self.data_dir_reduce
        else:
            directory = self.data_dir
        self.train_ds = CustomImageFolder(directory+'/train', target_transform=lambda x: self.image_label_mapping[x])
        self.test_ds = CustomImageFolder(directory+'/test', target_transform=lambda x: self.image_label_mapping[x])
        total_size = len(self.test_ds)
        val_size = int(0.50 * total_size)
        self.val_ds, self.test_ds = random_split(self.test_ds, [val_size, val_size]) 

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