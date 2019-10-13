import numpy as np
import pandas as pd
import os
import tensorflow as tf
import torch
from dataloader import Loader
from nn import simple_s2s , simple_s2s_lib , attention_s2s_lib
from keras.callbacks import EarlyStopping , TensorBoard , ModelCheckpoint , ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

#callbacks
ES = EarlyStopping(
  monitor = 'val_loss',
  patience = 5,
  verbose = 0,
  mode = 'auto'
)

MC = ModelCheckpoint(
  './logs_simple_s2s/weight.hdf5' , 
  monitor = 'val_loss',
  verbose = 0,
  save_best_only = True,
  save_weights_only = False,
  mode = 'auto',
  period = 2
)

TB = TensorBoard(
  log_dir = './logs_simple_s2s',
  histogram_freq = 1,
  write_graph = True,
  write_images = False
)

RP = ReduceLROnPlateau(
  monitor = 'val_loss',
  factor = 0.1,
  patience = 3,
  verbose = 0,
  mode = 'auto',
)

callbacks = [ES , TB , RP]


#create data
train_set = Loader(mode = 'train')
data_loader = torch.utils.data.DataLoader(train_set , batch_size = 300  , shuffle = False , num_workers = 1)
for i , (x , y) in enumerate(data_loader):
    x = x.data.numpy()
    y = y.data.numpy()

#main
if __name__ == '__main__': 
  model  = simple_s2s()
  model.fit(x , y , epochs = 2000  , batch_size = 1 , verbose = 1 , validation_split = 0.1 , callbacks = callbacks)
  
