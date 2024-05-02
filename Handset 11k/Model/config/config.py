from dataclasses import dataclass 

@dataclass
class Model:
    IN_CHANNELS : int
    NUM_CLASSES : int

@dataclass
class Train:
    LEARNING_RATE   :   float
    BATCH_SIZE      :   int
    NUM_EPOCHS      :   int
    PRECISION       :   str
    NUM_WORKERS     :   int
    ACCELERATOR     :   str
    REDUCE_DATASET  :   bool
    DATASET_SIZE    :   int

@dataclass
class DATASET:
    DATA_DIR : str

@dataclass
class Configuration:
    model : Model
    train : Train
    dataset : DATASET