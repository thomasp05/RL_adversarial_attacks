# this class contains the neural network models for the deep deterministic policy gradient algorithm
# it also contains the lenet5 model under attack for MNIST
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


class Lenet5(nn.Module): 
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(1, 6, 5) 
        self.S2 = nn.AvgPool2d(2, 2)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5) 
        self.S4 = nn.AvgPool2d(2) 
        self.F5 = nn.Linear(16*4*4, 120)
        self.F6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10) 

    def forward(self, x): 
        y = self.C1(x)
        y = F.relu(y)
        y = self.S2(y)

        y = self.C3(y)
        y = F.relu(y)
        y = self.S4(y)
     
        y = y.view(-1, 16*4*4)  
        y = self.F5(y)
        y = F.relu(y)

        y = self.F6(y)
        y = F.relu(y)

        y = self.output(y)

        return y

    def evaluate(self, valloader, device): 
        correct_count = 0 
        all_count = 0 
        for images, labels in valloader: 
            images, labels = images, labels
            for i in range(len(labels)): 
                image = images[i].view(1, 1, 28, 28).to(device)
                label = labels.numpy()[i]
                with(torch.no_grad()):
                    output = list(self(image).to("cpu").numpy()[0])
                
                    prediction = output.index(max(output))
                
                    if(prediction == label): 
                        correct_count += 1
                    all_count += 1

        return all_count, correct_count
