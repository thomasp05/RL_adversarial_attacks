# pour calculer la performance en validation du réseau LeNet 

import numpy as np 
from models import * 
from environment import environment
import torch 
import torchvision 
from torchvision import datasets, transforms 

# check if gpu available 
print("CUDA available: ", torch.cuda.is_available()) 
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 

# import MNIST Lenet5 model trained from MNIST_model.py 
lenet = Net()
state = torch.load('models/mnist_lenet.pt', map_location=device)
lenet.load_state_dict(state) 
lenet.to(device)
lenet.eval()

# import Mnist validation set 
transform = transforms.Compose([transforms.ToTensor(),])
valset = datasets.MNIST('data/mnist/validation', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, shuffle=True,  batch_size=1)

#test mnist lenet5 model performance 
all_count, correct_count = lenet.evaluate(valloader, device)
print("Number of Images Tested=", all_count) 
print("Model Accuracy =", (correct_count/all_count)*100)
