# this file is for testing the actor-critic model trained in main.py 


import numpy as np 
from models import * 
from environment import environment
import torch 
import torchvision 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np 
from copy import deepcopy
from ReplayBuffer import ReplayBuffer

pdf = PdfPages("images_test_label_9.pdf")

# check if gpu available 
print("CUDA available: ", torch.cuda.is_available()) 
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 

# import MNIST Lenet5 model trained from MNIST_model.py 
lenet = Net()
state = torch.load('models/mnist_lenet.pt', map_location=device)
# lenet = Net()
# state = torch.load("models/lenet_mnist_model.pth")
lenet.load_state_dict(state) 
lenet.eval()
lenet.to(device)

# import Mnist validation set 
transform = transforms.Compose([transforms.ToTensor(),])
valset = datasets.MNIST('data/mnist/validation', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, shuffle=True,  batch_size=1)

#test mnist lenet5 model performance 
# all_count, correct_count = lenet.evaluate(valloader, device)
# print("Number of Images Tested=", all_count) 
# print("Model Accuracy =", (correct_count/all_count)*100)


# get an image to attack from the valloader 
# dataiter = iter(valloader) 
# image, label = dataiter.next() 
# with(torch.no_grad()):
#     prediction = lenet.forward(image.to(device))
#     feature_map = lenet.featureMap(image.to(device)).squeeze()



########################
##### DPPG ATTACK ######
########################

# networks parameters 
input_size_actor = 320 + 20 # the article uses 300 but idk which model they use for MNIST classfication 
output_size_actor = 784
input_size_critic = input_size_actor + output_size_actor     # because the input is the concatenation of the context and action state vectors 
output_size_critic = 1
hidden_size = 512

# other parameters 
lr_critic = 0.0005                # learning rate for critic network 
lr_actor = 0.1 * lr_critic        # learning rate for actor network 
gamma = 0.9                      # discount factor 
tau = 0.1                         # parameter for soft-update
buffer_size = 500000                # replay buffer size (couldnt find the value for this parameter in the article)
batch_size = 32                 # batch size for training of actor and critic network (coundnt find the value for this parameter in the article) 

# init the replay buffer 
replay_buffer = ReplayBuffer(buffer_size, 42)

# init the adversarial environment
env = environment(lenet, valset, device, 42)

# Init the actor and critic models as well as their target models from the init class 
actor = Actor(input_size_actor, hidden_size, output_size_actor).to(device)                 
critic = Critic(input_size_critic, hidden_size, output_size_critic).to(device) 

# load trained state 
state_actor = torch.load('DDPG_models/targeted_attack_9.pt')
actor.load_state_dict(state_actor) 
actor.eval()
actor.eval()
epsilon = 1
target_label = 9

#### main loop ####
for episode in range(100):
    state = env.reset(target_label=target_label)
    reward = [] 
    loss = []
    episode_done = False 
    nb_iteration = 0
    max_iter = 0

    
    while not episode_done and max_iter < 100: 
        # compute the action to take with the actor network, which approximates the Q-function
        action = actor.forward(state.to(device))   

        # add noise to the action 
        noise = torch.rand(action.shape) * 0.1 
        action = action #+ noise.to(device)


        # take a step in the environment with the action chosen from the actor netork and observe new state and reward
        new_state, r, episode_done = env.step(action.detach(), nb_iteration) 
        reward.append(r)
        state = new_state

        # update the nb of iteration before episode_done
        nb_iteration += 1
        max_iter += 1
    epsilon =  epsilon = max(epsilon * 0.99, 0.1)

        


    print("Number of iterations required for misclassification : ", nb_iteration)
    print("We are at episode: ", episode)
    print("Total reward for the episode ", np.sum(reward))
    print(" ")

    if(episode_done): 
        predictions = lenet.forward(env.img.unsqueeze(0))
        # print(predictions)
        # print(torch.argmax(predictions))
        np_img = env.img.to("cpu").detach().view(28, 28)
        np_img1 = env.original_image.to("cpu").detach().view(28,28)
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np_img,  cmap="Greys")
        plt.title('Image générée')
        plt.subplot(2,2,2)
        plt.imshow(np_img1,  cmap="Greys")
        plt.title('Image originale')
        plt.show()
        classes = 'Classe réelle : ' + str(env.label) + ', classe prédite : ' + str(torch.argmax(predictions).to("cpu").numpy())
        plt.text(0.05, 0.05, classes, transform=fig.transFigure, size=12)
        pdf.savefig()
        plt.close()


pdf.close()
