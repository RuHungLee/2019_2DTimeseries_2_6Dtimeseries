import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import os
from dataloader import Loader
from nn import simple_s2s , simple_s2s_lib , attention_s2s_lib
from keras.callbacks import EarlyStopping , TensorBoard , ModelCheckpoint , ReduceLROnPlateau



#adjust gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
  './logs_0.001/weight.hdf5' , 
  monitor = 'val_loss',
  verbose = 1,
  save_best_only = True,
)

TB = TensorBoard(
  log_dir = './logs_0.001',
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

callbacks = [ES , MC , TB , RP]


#create data
train_set = Loader(mode = 'train')
data_loader = torch.utils.data.DataLoader(train_set , batch_size = 300  , shuffle = False , num_workers = 1)
for i , (x , y) in enumerate(data_loader):
    x = x.data.numpy()
    y = y.data.numpy()

#main
if __name__ == '__main__': 
  model  = attention_s2s_lib()
  model.fit(x , y , epochs = 2000  , batch_size = 2 , verbose = 1 , validation_split = 0.1 , callbacks = callbacks)
  
