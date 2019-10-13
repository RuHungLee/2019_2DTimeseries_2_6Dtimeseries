from keras.models import Sequential
from keras import optimizers
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense , RepeatVector

import numpy as np
import pandas as pd
from seq2seq.models import SimpleSeq2Seq , Seq2Seq , AttentionSeq2Seq
import tensorflow as tf
import os


#para
input_steps = 600
input_dim = 2
hidden_dim = 30
output_steps = 800
output_dim = 6
encoder_stack = 3
decoder_stack = 3


#define model
def simple_s2s():
  model = Sequential()
  model.add(LSTM(hidden_dim , input_shape = (input_steps , input_dim)))
  model.add(RepeatVector(output_steps))
  model.add(LSTM(hidden_dim , return_sequences = True))
  model.add(TimeDistributed(Dense(output_dim)))
  model.compile(loss = 'mse' , optimizer = 'adam')
  return model

def simple_s2s_lib():
  model  = Seq2Seq(output_dim = output_dim , hidden_dim = hidden_dim , output_length = output_steps , input_shape = (input_steps , input_dim) , peek = True , depth = (encoder_stack , decoder_stack) , bidirectional = True)
  model.compile(loss = 'mse' , optimizer = 'rmsprop')
  return model

def attention_s2s_lib():
  model = AttentionSeq2Seq(output_dim = output_dim , hidden_dim = hidden_dim , output_length = output_steps , input_shape = (input_steps , input_dim) , depth = (encoder_stack , decoder_stack) , bidirectional = True)
  sgd = optimizers.SGD(lr = 0.0001 , decay = 1e-6 , momentum = 0.9 , nesterov = False)
  model.compile(loss = 'mse' , optimizer = sgd)
  return model


#main
from keras.utils import plot_model
if __name__ == '__main__':

    model = attention_s2s_lib()
    plot_model(model , to_file = './model.png')
    #x = np.random.random((50 , 600 , 2))
    #y = np.random.random((50 , 800 , 6))
    #model.fit(x , y , epochs = 1 , batch_size = 10  , validation_split = 0.2)
