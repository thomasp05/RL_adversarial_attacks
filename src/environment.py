# this class is for the environment of the RL attack. 
# 
import numpy as np 
import torch 
from matplotlib import pyplot as plt
from torchvision import transforms

class environment: 
    def __init__(self, model, imgset, device, seed=None): 
        """
        model: neural net model of the environment under attack 
        imgset: image set 
        device: "cuda" or "cpu" 
        """
        self.model = model
        self.imgset = imgset
        self.nb_images = imgset.__len__() 
        self.device = device

        # fix the seed if provided
        np.random.seed(seed)
        

    def reset(self): 
        """
        reset returns a the convolutional feature map of a random image from the data loader,
        its model prediction and one hot encoded label vector
        """ 

        prediction = -1
        self.label = -2
        while(prediction != self.label):
             # compute random index for randomly selecting an image from the image set 
            index = np.random.randint(0, self.nb_images)

            # get the random image and compute its predicted class with the neural network 
            self.img, self.label = self.imgset.__getitem__(index) 

            # move img to device 
            self.img = self.img.to(self.device)
            self.original_image = self.img
            
            
            with torch.no_grad(): 
                self.prediction = self.model(self.img.view(1, 1, 28, 28)).squeeze() 
            prediction = torch.argmax(self.prediction)
        
        # compute the feature map 
        with torch.no_grad(): 
            self.feature_map = self.model.featureMap(self.img.view(1, 1, 28, 28)).squeeze() 

        # create the one-hot encoded vector for the label 
        # one_hot = np.zeros(self.prediction.shape)
        one_hot = torch.zeros(self.prediction.shape, device=self.device)
        one_hot[self.label] = 1
        self.one_hot = one_hot

        # format the context 
        state = torch.cat((self.feature_map.type(torch.DoubleTensor), self.prediction.type(torch.DoubleTensor), one_hot.type(torch.DoubleTensor)), 0)
        self.current_state = state.type(torch.FloatTensor)


        # counter for episode done 
        self.counter = 0 

        return self.current_state
        

    def step(self, action): 
        """
        action:  vector containing image perturbations 
        """
        # compute the predicted class for the given action with the neural network
        # reformat the action into [1, 1, 28, 29] 
        # action_ = action.view(-1, 28, 28).unsqueeze(0) 

        # normalize action between -1 and 1 like original image under attack
        # action_ = (action_ - action_.min()) /(action_.max() - action_.min()) * (-2) + 1

        new_image = action.view(-1, 28, 28)
        new_image = new_image * 1  + self.img  # le probleme est ici, je crois que jai besoin de update self.image et je dois rajouter self.currentimage (image modifiee au fil des iterationa)
        
        # add noise to perturbation
      
        noise =  torch.rand(self.img.shape)
        new_image = new_image #+ noise.to(self.device) 
        action_ = new_image


        # normalize the new image 
        norm = transforms.Compose([transforms.Normalize((0.5,), (0.5,)),])
        new_image = norm(new_image)
        action_ = norm(action_)
        
        action_ = action_.unsqueeze(0)

        print(action_.max()) 
        print(action_.min())
        print(new_image.max()) 
        print(new_image.min())
        print(" ")
        # display the image 
        np_img = new_image.to("cpu").detach().view(28, 28)
        img2 = action_.squeeze(0).to("cpu").detach().view(28, 28)
        plt.subplot(1,2,1)
        plt.imshow(np_img,  cmap="gray")
        plt.subplot(2,2,2)
        plt.imshow(img2,  cmap="gray")
        plt.show()

        # # compute the new predictions and convolutional features map
        with torch.no_grad(): 
            new_prediction = self.model.forward(action_).squeeze() 
            new_feature_map = self.model.featureMap(action_).squeeze()

        next_state = torch.cat((new_feature_map.type(torch.DoubleTensor), new_prediction.type(torch.DoubleTensor), self.one_hot.type(torch.DoubleTensor)), 0).type(torch.FloatTensor)

        #check if image is misclassified and compute reward 
        episode_done = False
       
        ######################
        ### calcul du reward # 
        ######################
        temp = self.prediction[self.label].to("cpu").numpy()
        temp2 = new_prediction[self.label].to("cpu").numpy()

        # pour predicions des autres classes 
        pred_reward = 0
        for i, elem in enumerate(new_prediction): 
            if(i != self.label):
                old = self.prediction[i].to("cpu").numpy()
                new = elem.to("cpu").numpy()
                temp3 = new - old 
                pred_reward += temp3

        # difference btw feature map and new feature map
        feature_reward = self.feature_map - new_feature_map
        feature_reward = feature_reward.to("cpu").numpy().sum()

        # difference btw image and new image 
        img_reward = self.original_image.squeeze() - new_image.squeeze()
        img_reward = img_reward.to("cpu").numpy().sum()

        # compute total reward 
        reward =  np.abs(temp - temp2) - 0.1 * np.abs(img_reward) + pred_reward
     
        
        # check if episode is done 
        if(torch.argmax(new_prediction) != self.label): 
               
            self.counter += 1
            if(self.counter >= 1): 
                episode_done = True 
                reward = reward + 100 #1000
                
            # # display the image 
            # np_img = new_image.to("cpu").detach().view(28, 28)
            # np_img1 = self.original_image.to("cpu").detach().view(28,28)
            # plt.subplot(1,2,1)
            # plt.imshow(np_img,  cmap="Greys")
            # plt.subplot(2,2,2)
            # plt.imshow(np_img1,  cmap="Greys")
            # plt.show()

            
        # update prediction and feature map before exiting
        self.prediction = new_prediction
        self.feature_map = new_feature_map 
        self.img = action_ .squeeze(0)

        return next_state, reward, episode_done

    