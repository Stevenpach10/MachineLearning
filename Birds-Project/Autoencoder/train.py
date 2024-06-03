from dataset import ImageDataModule
from model import Autoencoder
from config.config import Configuration
import os
import hydra
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid



@hydra.main(config_path="config", config_name="config")
def main(config : Configuration):
    data_dir = os.path.join(os.getcwd(), config.dataset.DATA_DIR)
    data_module = ImageDataModule(data_dir=data_dir, num_workers=config.train.NUM_WORKERS, batch_size=config.train.BATCH_SIZE)
 
    checkpoint_filename = "autoencoder_checkpoint.ckpt"
    checkpoint_path = os.path.join(config.dataset.BASE_DIR, checkpoint_filename)
    
    model = Autoencoder(input_channels=config.model.INPUT_CHANNELS, num_classes=config.model.NUM_CLASSES, learning_rate=config.train.LEARNING_RATE)

    if os.path.exists(checkpoint_path):
        model = Autoencoder.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                 input_channels=config.model.INPUT_CHANNELS, 
                                                 num_classes=config.model.NUM_CLASSES, 
                                                 learning_rate=config.train.LEARNING_RATE)
        print("Loaded checkpoint from", checkpoint_path)

    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.dataset.BASE_DIR,
        filename='autoencoder_checkpoint',
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )

    # Initialize the W&B logger
    wandb_logger = WandbLogger(
        project=config.wandb.PROJECT_NAME,
        log_model=config.wandb.LOG_MODEL
    )

    trainer = pl.Trainer(max_epochs=config.train.NUM_EPOCHS,  
                         accelerator= config.train.ACCELERATOR,
                         callbacks=[checkpoint_callback],
                         logger=wandb_logger)
    if config.train.FIT:
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()