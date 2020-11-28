# this file is for testing the actor-critic model trained in main.py 


import numpy as np 
from models import * 
from environment import environment
import torch 
import torchvision 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
import numpy as np 
from copy import deepcopy
from ReplayBuffer import ReplayBuffer


# import MNIST Lenet5 model trained from MNIST_model.py 
lenet = Net()
state = torch.load('models/mnist_lenet.pt')
# lenet = Net()
# state = torch.load("models/lenet_mnist_model.pth")
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
state_actor = torch.load('DDPG_models/actor.pt')
state_critic = torch.load('DDPG_models/critic.pt')
actor.load_state_dict(state_actor) 
critic.load_state_dict(state_critic) 
actor.eval()
actor.eval()


#### main loop ####
cumul_reward = [] 
cumul_loss = [] 
epsilon = 1

for episode in range(100): 
    state = env.reset()
    reward = [] 
    loss = []
    episode_done = False 
    nb_iteration = 0
    max_iter = 0

    
    while not episode_done and max_iter < 100: 
        # compute the action to take with the actor network, which approximates the Q-function
        # with torch.no_grad():
        action = actor.forward(state.to(device))    # TODO: add noise 

        # add noise to the action 

        noise = torch.rand(action.shape) /(nb_iteration+1) #* epsilon 
        action = action + noise.to(device) * 0.1

        # take a step in the environment with the action chosen from the actor netork and observe new state and reward
        new_state, r, episode_done = env.step(action.detach(), epsilon, nb_iteration) 
        reward.append(r)

        # save observations in the replay buffer 
        replay_buffer.store((state, action.to("cpu").detach(), r, new_state)) 
        state = new_state

        # check if enough samples stored in replay buffer 
        if(replay_buffer.buffer_len > batch_size): 

            if(nb_iteration % 1 == 0):
                # randomly sample a minibatch from the replay buffer 
                minibatch = replay_buffer.get_batch(batch_size) 

                # Unpack minibatch 
                states_batch = np.array([x[0].numpy() for x in minibatch]) 
                actions_batch = np.array([x[1].numpy() for x in minibatch]) 
                rewards_batch = np.array([x[2] for x in minibatch]) 
                next_states_batch = np.array([x[3].numpy() for x in minibatch]) 
                
                states_batch = torch.FloatTensor(states_batch)
                actions_batch = torch.FloatTensor(actions_batch)
                next_states_batch = torch.FloatTensor(next_states_batch)
                rewards_batch = torch.FloatTensor(rewards_batch) 

                # compute predicted q values           
                q_values = critic.forward(states_batch.to(device), actions_batch.to(device))

                # 1- compute next actions with actor target network 
                next_actions = actor_target(next_states_batch.to(device))

                # compute target q values with the Bellman equation
                # rewards_batch doesnt have the right dimensions 
                y_i =  rewards_batch + gamma * critic_target(next_states_batch.to(device), next_actions.detach().to(device)).to("cpu").squeeze()
                y_i =  y_i.unsqueeze(1)
        

                # compute loss for both actor and critic networks 
                loss_critic = critic_criterion(q_values.to(device), y_i.to(device)) 
                loss_actor = -critic.forward(states_batch.to(device), actor.forward(states_batch.to(device)).to(device)).mean()
                loss.append(loss_critic)

                # update actor network (for policy approximation)
                actor_optim.zero_grad()
                loss_actor.backward()
                actor_optim.step() 

                # update critic network (for Q function approximation)
                critic_optim.zero_grad() 
                loss_critic.backward() 
                critic_optim.step() 

                # perform soft update to update the weights of the target networks 
                # update target networks 
                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
        
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
     
        
        # update the nb of iteration before episode_done
        nb_iteration += 1
        max_iter += 1
        epsilon =  epsilon = max(epsilon * 0.9, 0.1)
        
    # save information to asses performance after 
    cumul_reward.append(np.sum(reward))
    cumul_loss.append(np.sum(loss))


    print("Number of iterations required for misclassification : ", nb_iteration)
    print("We are at episode: ", episode)
    print("Total reward for the episode ", np.sum(reward))
    print(" ")

    if(episode_done): 
        # predictions = lenet.forward(env.original_image.unsqueeze(0))
        # print(predictions)
        np_img = env.img.to("cpu").detach().view(28, 28)
        np_img1 = env.original_image.to("cpu").detach().view(28,28)
        plt.subplot(1,2,1)
        plt.imshow(np_img,  cmap="gray")
        plt.subplot(2,2,2)
        plt.imshow(np_img1,  cmap="gray")
        plt.show()
