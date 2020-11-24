# this class is for the environment of the RL attack. 
# 
import numpy as np 
import torch 
from matplotlib import pyplot as plt

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

        # compute random index for randomly selecting an image from the image set 
        index = np.random.randint(0, self.nb_images)

        # get the random image and compute its predicted class with the neural network 
        self.img, self.label = self.imgset.__getitem__(index) 
        prediction = -1
        while(prediction != self.label):
            with torch.no_grad(): 
                self.prediction = self.model.forward(self.img.unsqueeze(0).to(self.device)).squeeze()
            prediction = torch.argmax(self.prediction)
        
        # compute the feature map 
        with torch.no_grad(): 
            self.feature_map = self.model.featureMap(self.img.to(self.device).unsqueeze(0)).squeeze()

        # create the one-hot encoded vector for the label 
        one_hot = np.zeros(self.prediction.shape)
        one_hot[self.label] = 1
        one_hot = torch.tensor(one_hot)  
        self.one_hot = one_hot

        # format the context 
        state = torch.cat((self.feature_map.type(torch.DoubleTensor), self.prediction.type(torch.DoubleTensor), one_hot.type(torch.DoubleTensor)), 0)
        self.current_state = state.type(torch.FloatTensor)
        return self.current_state
        

    def step(self, action): 
        """
        action:  vector containing image perturbations 
        """
        # compute the predicted class for the given action with the neural network
        # reformat the action into [1, 1, 28, 29] 
        action_ = action.view(-1, 28, 28).unsqueeze(0) 
        # print(action_.shape) 

        new_image = action.view(-1, 28, 28)
        new_image = new_image + self.img.to(self.device)    # le probleme est ici, je crois que jai besoin de update self.image et je dois rajouter self.currentimage (image modifiee au fil des iterationa)
        action_ = new_image.unsqueeze(0)

        # display the image 
        # np_img = new_image.to("cpu").detach().view(28, 28)
        # plt.imshow(np_img,  cmap="gray")
        # plt.show()

        # compute the new predictions and convolutional features map
        with torch.no_grad(): 
            self.prediction = self.model.forward(action_).squeeze() 
            self.feature_map = self.model.featureMap(action_).squeeze()

        next_state = torch.cat((self.feature_map.type(torch.DoubleTensor), self.prediction.type(torch.DoubleTensor), self.one_hot.type(torch.DoubleTensor)), 0).type(torch.FloatTensor)

        #check if image is misclassified and compute reward 
        episode_done = False
        reward = 0 

        # print(self.prediction)
        # print(torch.argmax(self.prediction))
        # print(self.label)
        if(torch.argmax(self.prediction) != self.label): 
            episode_done = True 
            reward = 1

        return next_state, reward, episode_done

    