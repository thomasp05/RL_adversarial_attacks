# this class contains the neural network models for the deep deterministic policy gradient algorithm

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.autograd 
import torch.optim as optim 
import numpy as np 


class Actor(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, output_size) 

    def forward(self, state): 
        '''
        state: 
        action: 
        '''
        x = state # concatenate because the input is both 
        x = F.relu(self.linear1(x)) 
        x = F.relu(self.linear2(x)) 

        return x



class Critic(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, output_size) 

    def forward(self, state, action): 
        '''
        state: 
        action: 
        '''
        x = torch.cat([state, action], 1) # concatenate because the input is both 
        x = F.relu(self.linear1(x)) 
        x = F.relu(self.linear2(x)) 

        return x
