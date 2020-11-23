#This is where the RL attack is performed

import numpy as np 
from models import * 
from environment import environment
import torch 
import torchvision 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
import numpy as np 


#parameters 
input_size_actor = 256 + 20 # the article uses 300 but idk which model they use for MNIST classfication 
output_size_actor = 784
input_size_critic = input_size_actor + output_size_actor     # because the input is the concatenation of the context and action state vectors 
output_size_critic = 1
hidden_size = 512

# import MNIST Lenet5 model trained from MNIST_model.py 
lenet = Lenet5()
state = torch.load('models/mnist_lenet.pt')
lenet.load_state_dict(state) 
lenet.eval()

# check if gpu available 
print("CUDA available: ", torch.cuda.is_available()) 
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 
lenet.to(device)


# import Mnist validation set 
transform = transforms.Compose([transforms.ToTensor(),])
valset = datasets.MNIST('data/mnist/validation', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, shuffle=True,  batch_size=1)

# Init the actor and critic models as well as their target models from the init class 
actor = Actor(input_size_actor, hidden_size, output_size_actor) 
critic = Critic(input_size_critic, hidden_size, output_size_critic) 
actor.to(device)
critic.to(device)

#test mnist lenet5 model performance 
# all_count, correct_count = model.evaluate(valloader, device)
# print("Number of Images Tested=", all_count) 
# print("Model Accuracy =", (correct_count/all_count)*100)


# helper function for displaying an image 
def imshow(img):
    # img = img / 2 + 0.5   #unnormalize the image 
    img = img.squeeze()
    npimg = img.squeeze().numpy()
    plt.imshow(img,  cmap="gray")
    plt.show()

# get an imlaage to attack from the valloader 
dataiter = iter(valloader) 
image, label = dataiter.next() 
with(torch.no_grad()):
    prediction = lenet.forward(image.to(device))
    feature_map = lenet.featureMap(image.to(device)).squeeze()

# Concatenate the feature_map, labels and prediction for the input to the actor network 
one_hot = np.zeros(prediction.shape[1])
one_hot[label] = 1
one_hot = torch.tensor(one_hot)
prediction = prediction.squeeze()
input = torch.cat((feature_map.type(torch.DoubleTensor), prediction.type(torch.DoubleTensor), one_hot.type(torch.DoubleTensor)), 0)

# test the actor network 
input = input.type(torch.FloatTensor)
temp = actor.forward(input.to(device))

# test the critic network 
temp2 = critic.forward(input.to(device), temp.to(device))

# initialize the environment 
env = environment(lenet, valset, device) 
state = env.reset()
action = actor.forward(state.to(device)) 
q_val = critic.forward(state.to(device), action)
next_s, r, episode_done = env.step(action)


#perform the attack 

# TO-DO: define the cost function 
# impovements: use more powerfull neural nets for actor and critic networks