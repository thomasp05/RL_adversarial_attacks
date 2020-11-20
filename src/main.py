#This is where the RL attack is made 

import numpy as np 
from models import * 
import torch 
import torchvision 
from torchvision import datasets, transforms 


#transformations we want to perform before using the data in the pipeline 
transform = transforms.Compose([transforms.ToTensor(), 
                                ])

# import Mnist validation set 
# dowload the MNIST test set 
transform = transforms.Compose([transforms.ToTensor(),])
valset = datasets.MNIST('data/mnist/validation', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, shuffle=True,  batch_size=1)