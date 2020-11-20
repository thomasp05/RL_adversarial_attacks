#This is where the RL attack is performed

import numpy as np 
from models import * 
import torch 
import torchvision 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
import numpy as np 


#parameters 
input_size_actor = 300
output_size_actor = 784
input_size_critic = input_size_actor + output_size_actor      # because the input is the concatenation of the context and action state vectors 
output_size_critic = 1

# import MNIST Lenet5 model trained from MNIST_model.py 
model = Lenet5()
state = torch.load('models/mnist_lenet.pt')
model.load_state_dict(state) 
model.eval()

# check if gpu available 
print("CUDA available: ", torch.cuda.is_available()) 
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 
model.to(device)


# import Mnist validation set 
transform = transforms.Compose([transforms.ToTensor(),])
valset = datasets.MNIST('data/mnist/validation', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, shuffle=True,  batch_size=1)

#test mnist lenet5 model performance 
# all_count, correct_count = model.evaluate(valloader, device)
# print("Number of Images Tested=", all_count) 
# print("Model Accuracy =", (correct_count/all_count)*100)


# Init the actor and critic models as well as their target models from the init class 
# actor = Actor()




#perform the attack 