#This is where the RL attack is performed

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

# set numpy seed 
np.random.seed(42)
torch.manual_seed(42)

# check if gpu available 
print("CUDA available: ", torch.cuda.is_available()) 
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 


# import MNIST Lenet5 model trained from MNIST_model.py 
lenet = Net()
state = torch.load('models/mnist_lenet.pt', map_location=device)
# lenet = Net()
# state = torch.load("models/lenet_mnist_model.pth", map_location=device)
lenet.load_state_dict(state) 
lenet.eval()
lenet.to(device)

# import Mnist validation set 
transform = transforms.Compose([transforms.ToTensor(),])
valset = datasets.MNIST('data/mnist/validation', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, shuffle=True,  batch_size=1)

# helper function for displaying an image 
def imshow(img):
    img = img.squeeze()
    npimg = img.squeeze().numpy()
    plt.imshow(img,  cmap="Greys")
    plt.show()


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
buffer_size = 50000                # replay buffer size (couldnt find the value for this parameter in the article)
batch_size = 32                 # batch size for training of actor and critic network (coundnt find the value for this parameter in the article) 


# loop through all classes 
for i in range(10):
    
    # important variables 
    cumul_reward = [] 
    cumul_loss = [] 
    nb_queries = []
    cumul_norm = []
    original_labels = []
    predicted_labels = []
    epsilon = 1
    epsilon_decay = 0.99
    epsilon_min = 0.1
    nb_episode = 100
    successful_attacks = 0

    # the label to be missclassified
    target_label = i

    fileName = "One_NN_per_Class_Results/class_negr5_1000/results/model_for_class{0}_{1}.pdf".format(target_label, nb_episode)
    modelName = 'One_NN_per_Class_Results/class_negr5_1000/models/model_for_class{0}_{1}.pt'.format(target_label, nb_episode)

    # name of the textfile where the results will be saved
    textFileName = "One_NN_per_Class_Results/class_negr5_1000/results_{0}.txt".format(nb_episode)

    # name for plots 
    reward_plot_name = "One_NN_per_Class_Results/class_negr5_1000/plots/cumul_reward_class{0}_{1}.png".format(target_label, nb_episode)
    loss_plot_name = 'One_NN_per_Class_Results/class_negr5_1000/plots/cumul_loss_class{0}_{1}.png'.format(target_label, nb_episode)

    # open the pdf to save the images 
    pdf = PdfPages(fileName)

    # init the replay buffer 
    replay_buffer = ReplayBuffer(buffer_size, 42)

    # init the adversarial environment
    env = environment(lenet, valset, device, 42)

    # Init the actor and critic models as well as their target models from the init class 
    actor = Actor(input_size_actor, hidden_size, output_size_actor).to(device)

    ### Uncomment the following lines to use transfer learning with a model trained on the dataset using all classes
    #state_actor = torch.load('DDPG_models/actor_base.pt')
    #actor.load_state_dict(state_actor)

    critic = Critic(input_size_critic, hidden_size, output_size_critic).to(device) 
    actor_target = deepcopy(actor).to(device)
    critic_target = deepcopy(critic).to(device) 

    # init the model loss funcitions and optimizers 
    critic_criterion = nn.MSELoss()                           # actor_loss is not defined here because it is computed using the critic network according to the DDPG algorithm
    actor_optim = optim.Adam(actor.parameters(), lr_actor) 
    critic_optim = optim.Adam(critic.parameters(), lr_critic) 


    #### main loop ####
    for episode in range(nb_episode): 
        state = env.reset(target_label=target_label)
        reward = [] 
        loss = []
        episode_done = False 
        nb_iteration = 0
        max_iter = 0


        while not episode_done and max_iter <100: 
            # compute the action to take with the actor network, which approximates the Q-function
            action = actor.forward(state.to(device))  

            # add noise to the action 
            noise = torch.rand(action.shape) * epsilon
            action = action + noise.to(device)

            # take a step in the environment with the action chosen from the actor netork and observe new state and reward
            new_state, r, episode_done = env.step(action.detach(), nb_iteration) 
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
                    next_actions = actor_target.forward(next_states_batch.to(device))

                    # compute target q values with the Bellman equation
                    y_i =  rewards_batch + gamma * critic_target.forward(next_states_batch.to(device), next_actions.detach().to(device)).to("cpu").squeeze()
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
        epsilon =  epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # save information to asses performance after 
        cumul_reward.append(np.sum(reward))
        cumul_loss.append(np.sum(loss))
        nb_queries.append(nb_iteration)
        original_labels.append(env.label)

        if episode_done: 
            # increment counter 
            successful_attacks += 1

            # get the prediction of the target network and the original and perturbed images
            predictions = lenet.forward(env.img.unsqueeze(0))
            perturbed_image = env.img.to("cpu").detach().view(28, 28)
            original_image = env.original_image.to("cpu").detach().view(28,28)

            perturbation = perturbed_image - original_image
            l2_norm = np.linalg.norm(perturbation) 
            cumul_norm.append(l2_norm) 
            pred_label = torch.argmax(predictions).to("cpu").detach().numpy().item()
            predicted_labels.append(pred_label)

            fig = plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(perturbed_image,  cmap="Greys")
            plt.title('Image générée')
            plt.subplot(2,2,2)
            plt.imshow(original_image,  cmap="Greys")  
            plt.title('Image originale')
            classes = 'Classe réelle : ' + str(env.label) + ', classe prédite : ' + str(torch.argmax(predictions).to("cpu").numpy())
            plt.text(0.05, 0.05, classes, transform=fig.transFigure, size=12)
            pdf.savefig()
            plt.close()
        else:
            predicted_labels.append(env.label)


    # print results
    print("Average nb of queries : ", np.sum(nb_queries) / nb_episode)
    print("Accuracy", successful_attacks / nb_episode)
    print("Average norm ", np.sum(cumul_norm) / successful_attacks)

    # save the actor model and close pdf
    pdf.close()
    state_actor = actor.state_dict()
    torch.save(state_actor, modelName)

    # plot cumulative reward and loss and save to png  
    plt.figure()
    plt.plot(cumul_reward)
    plt.xlabel('Episodes')
    plt.ylabel('Reward cumulatif')
    plt.title('Reward cumulatif par épisode')
    plt.savefig(reward_plot_name)

    plt.figure()
    plt.plot(cumul_loss) 
    plt.xlabel('Episodes')
    plt.ylabel('Loss cumulative')
    plt.title('Loss cumulative par épisode')
    plt.savefig(loss_plot_name)


    # last step is to test the generalisation performance of the model 
    # load trained state 
    state_actor = torch.load(modelName)
    actor.load_state_dict(state_actor) 
    actor.eval()
    success = 0

    for episode in range(nb_episode):
        state = env.reset()
        reward = [] 
        loss = []
        episode_done = False 
        nb_iteration = 0
        max_iter = 0

        while not episode_done and max_iter < 100: 
            # compute the action to take with the actor network, which approximates the Q-function
            action = actor.forward(state.to(device))   

            # take a step in the environment with the action chosen from the actor netork and observe new state and reward
            new_state, r, episode_done = env.step(action.detach(), nb_iteration) 
            reward.append(r)
            state = new_state

            # update the nb of iteration before episode_done
            nb_iteration += 1
            max_iter += 1

        if(episode_done): 
            success += 1

    # print results 
    print("Performance en généralisation : {0}\n".format(success/nb_episode))

    # save results to a text file
    with open(textFileName, "a") as text_file:
        text_file.write("Target class : {0} \n".format(i))
        text_file.write("Average nb of queries : {0}\n".format(np.sum(nb_queries) / nb_episode))
        text_file.write("Accuracy : {0}\n".format(successful_attacks / nb_episode))
        text_file.write("Average norm : {0}\n".format(np.sum(cumul_norm) / successful_attacks))
        text_file.write("Original class: {0}\n".format(str(original_labels)))
        text_file.write("Predicted class : {0}\n".format(str(predicted_labels)))
        text_file.write("Performance en généralisation : {0}\n".format(success/nb_episode))
        text_file.write("\n")
