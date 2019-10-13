#import lib
import torch
import torch.nn as nn
import torch.nn.utils as utils
import os
import numpy as np
from dataloader import Loader
from nn import Model
from visdom import Visdom

#para
epochs = 2000
lr = 1e-5
batch_size = 2
device = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = device

#main
if __name__ == '__main__':

    #create model
    model = Model()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters() , lr)
    loss = torch.nn.MSELoss()
    visdom_server = Visdom(port=8097)
    

    #create dataloade
    train_set = Loader(mode = 'train')
    val_set = Loader(mode = 'test') 
    data_loader = torch.utils.data.DataLoader(train_set , batch_size = batch_size , shuffle = True , num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_set , batch_size = batch_size , shuffle = True , num_workers=1)
    
    #training
    print(f'The number of training data {len(data_loader)}')
    print(f'The number of testing data {len(val_data_loader)}')


    for epoch in range(epochs):

        #training period  
        total_loss = 0
        for i , (twod , sixd) in enumerate(data_loader):
            twod , sixd  =  twod.cuda() , sixd.cuda()
            pred = model(twod)
            mse = loss(pred , sixd)  
            optimizer.zero_grad()
            mse.backward()
            optimizer.step()
            total_loss += mse.item()
        total_loss /= len(data_loader)
        print(f'training epoch:{epoch}\tloss:{total_loss}')
        visdom_server.line([total_loss] , [epoch] , win = 'loss' , env='2d_to_6d' , update = 'append')
        

        #validation period 
        total_loss = 0
        for i , (twod , sixd) in enumerate(val_data_loader):
            twod , sixd  =  twod.cuda() , sixd.cuda()
            pred = model(twod)
            mse = loss(pred , sixd) 
            total_loss += mse.item()
        total_loss /= len(val_data_loader)
        print(f'val epoch:{epoch}\tloss:{total_loss}')
        visdom_server.line([total_loss] , [epoch] , win = 'val_loss' , env='2d_to_6d' , update = 'append')
        
        





