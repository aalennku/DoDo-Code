import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import os

def load_embed_model(args):
    embed_model = Embed_model(input_channels = 5, length = args.padded_length, 
                              output_dim = args.dimension, conv_channels = 256).to(args.device)
    embed_model_path = os.path.join(args.save_path,'embed_model.pth')
    embed_model.load_state_dict(torch.load(embed_model_path,map_location=torch.device('cuda:{}'.format(args.gpu))))
    embed_model.eval()
    return embed_model

class Embed_model(torch.nn.Module):
    def __init__(self, input_channels = 5, length = 160, output_dim = 40, conv_channels = 64):
        super(Embed_model, self).__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.length = length
        self.channels = conv_channels
        
        self.conv = nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.fc1 = torch.nn.Linear(self.length*self.channels,  output_dim*4)
        self.fc2 = torch.nn.Linear(output_dim*4, output_dim)
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01,eps=1e-9,affine=False)
        self.scaling = nn.Parameter(torch.Tensor([np.math.sqrt(40/output_dim)]), requires_grad=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final_bn(x)
        x = x * self.scaling
        return x
    
class Twin_model(torch.nn.Module):
    def __init__(self, embed_model, output_dim):
        super(Twin_model, self).__init__()
        self.encoder = embed_model
        self.output_dim = output_dim
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01,eps=1e-9,affine=False)

    def forward(self, x):
        x, y = torch.unbind(x, dim=1)
        x_embed = self.encoder(x)
        y_embed = self.encoder(y)
        distance = torch.sum((x_embed-y_embed)**2,dim=-1)
        return distance