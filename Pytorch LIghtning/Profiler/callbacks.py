from lightning import LightningModule, Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_start(trainer, pl_module)
        print('Starting to train!, please await!')

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_end(trainer, pl_module)
        print('Training is done.')
        
