# this class is for the environment of the RL attack. 
# 
import numpy as np 
import torch 

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
                self.prediction = self.model.forward(self.img.unsqueeze(0).to(self.device)).to("cpu").squeeze()
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
        with torch.no_grad(): 
            new_prediction = self.model.forward(action_).squeeze() 
            new_feature_map = self.model.featureMap(action_).squeeze()

        self.feature_map = new_feature_map
        self.prediction = new_prediction

        next_state = torch.cat((self.feature_map.type(torch.DoubleTensor), self.prediction.type(torch.DoubleTensor), self.one_hot.type(torch.DoubleTensor)), 0)

        # compute reward 
        reward = 1
        
        #check if image is misclassified 
        episode_done = False 

        return next_state, reward, episode_done

    