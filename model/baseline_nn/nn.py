import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet50
# from torchvision.models.resnet import resnet50

#parameter
input_layer = 1200
hidden_layer = 3000
output_layer = 800*6
#Model
class Model(nn.Module):

    def __init__(self, input_layer = input_layer , hidden_layer = hidden_layer , output_layer = output_layer):

        super().__init__()
        self.fc1 = nn.Linear(input_layer,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer,hidden_layer)
        self.fc3 = nn.Linear(hidden_layer,output_layer)       
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
    
        x = x.view(-1 , 1200)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = x.view(-1 , 800 , 6)    
        
        return x
