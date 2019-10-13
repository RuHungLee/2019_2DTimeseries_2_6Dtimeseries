import  sys
from PIL import  Image
import  numpy as np
import  pandas as pd
import  os
import  torch
from torchvision.transforms.transforms import ToTensor
from torch.utils.data import  Dataset

np.set_printoptions(threshold=sys.maxsize)

def loader(path , mode='2D'):

  if mode == '2D':
    cor = pd.read_csv(path , sep='\t' , header = None)
    cor = cor.to_numpy()
    cor = np.pad(cor , [(0, 600-cor.shape[0]) , (0  , 0)] , mode='constant' , constant_values = 0)
    cor = torch.from_numpy(cor)
    cor = cor.type(torch.float32)
    return cor

  elif mode == '6D':
    cor = pd.read_csv(path , sep=' ' , header = None)
    cor = cor.iloc[: , 2:8].to_numpy()
    cor = np.pad(cor , [(0, 800-cor.shape[0]) , (0  , 0)] , mode='constant' , constant_values = 0)
    cor = torch.from_numpy(cor)
    cor = cor.type(torch.float32)
    return cor

def readfile(pair_file):
  seq = []
  with open(pair_file , 'r') as f:
    lines = f.readlines()
    for line in lines:
      twod = line.split()[0]
      sixd = line.split()[1]
      seq.append((twod , sixd))
    return seq
  
class Loader(Dataset):
  def __init__(self , mode = 'train' , loader = loader):
    self.mode = mode
    self.loader = loader
    if self.mode == 'train':
        self.cor_path = readfile('/home/eric123/2D_to_6D_model/filename/train_pair.txt')
    elif self.mode == 'test':
        self.cor_path = readfile('/home/eric123/2D_to_6D_model/filename/test_pair.txt')

  def __len__(self):
    return len(self.cor_path)


  def __getitem__(self , idx):
    twod_path , sixd_path  = self.cor_path[idx]
    twod = self.loader(twod_path , mode = '2D')
    sixd = self.loader(sixd_path , mode = '6D')
    return twod , sixd 
    
if __name__ == "__main__":
    train_set = Loader(mode = 'train')
    data_loader = torch.utils.data.DataLoader(train_set , batch_size = 300 , shuffle = False , num_workers = 1)
    print('dataset num is: ',len(data_loader))
    for i  , (twod , sixd) in enumerate(data_loader):
        print(twod.shape)
        print(sixd.shape)
        print(twod.data.numpy().shape)
        print(sixd.data.numpy().shape)
        exit(-1)


