from data.dataset import *
#from dataset import *
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def sinFunction():
    x = np.linspace(0.0, 2.0 * np.pi, 800).reshape(-1, 1)
    y = np.sin(x)

    return x , y

#hyperparameters
batch_size = 32
validate_every_no_of_batches = 300
epochs = 10000
input_size = 1
output_size = 1
hidden_shapes = [100, 100]
lr = 0.02
has_dropout=False
dropout_perc=0.5
output_log = r"runs/sin_log.txt"

x, y = sinFunction()
data = dataset(x, y, batch_size)
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.6, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
