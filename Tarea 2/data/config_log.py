from data.dataset import *
#from dataset import *
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def sinFunction():
    x = np.linspace(0.0, 2.0 * np.pi, 500).reshape(-1, 1)
    y = np.log(x)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x = x_scaler.fit_transform(x)
    y = y_scaler.fit_transform(y)
    return x , y

#hyperparameters
batch_size = 25
validate_every_no_of_batches = 8
epochs = 1000
input_size = 1
output_size = 1
hidden_shapes = [50]
lr = 0.0085
has_dropout=True
dropout_perc=0.5
output_log = r"runs/sin_log.txt"

x, y = sinFunction()
data = dataset(x, y, batch_size)
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.6, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
