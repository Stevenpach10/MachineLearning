from dataclasses import dataclass

@dataclass
class Model:
    INPUT_SIZE  : int
    NUM_CLASSES : int

@dataclass
class Train:
    LEARNING_RATE   :   float
    BATCH_SIZE      :   int
    NUM_EPOCHS      :   int
    PRECISION       :   str
    NUM_WORKERS     :   int
    ACCELERATOR     :   str
    LOG_EVERY_STEPS :   int

@dataclass
class Dataset:
    DATA_DIR : str

@dataclass
class Configuration:
    MODEL : Model
    TRAIN : Train
    DATASET : Dataset
