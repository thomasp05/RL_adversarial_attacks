# this class is for the environment of the RL attack. 
 
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
        

    def reset(self, target_label=None, attack_label=None):
        """
        target_label: label of the images to be returned, if None it can be any label
        attack_label: label of the targeted class for the targeted attack
        reset returns a the convolutional feature map of a random image from the data loader,
        its model prediction and one hot encoded label vector
        """ 
        prediction = -1
        self.label = -2
        while(prediction != self.label or prediction==attack_label):
            while True:
                # compute random index for randomly selecting an image from the image set
                index = np.random.randint(0, self.nb_images - 1) 

                # get the random image and compute its predicted class with the neural network
                self.img, self.label = self.imgset.__getitem__(index)
        
                if(target_label is None or self.label == target_label):
                    break
        
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

        return self.current_state
        

    def step(self, action, t, target_class=-1): 
        """
        action:  vector containing image perturbations 
        """
        
        perturbation = action.view(-1, 28, 28) 
        new_image = self.img + perturbation * 0.1  # 0.1 cest la constante qui determine la force de la perturbation sur l'image

        # normalize the new image before passing it to the MNIST network 
        new_image = torch.clamp(new_image, 0, 1)   # Clamp pour garder dans le range [0,1]
 
        # compute the perturbation after clamping 
        perturbation = new_image - self.original_image

        # compute the new predictions and convolutional features map
        with torch.no_grad(): 
            new_prediction = self.model.forward(new_image.unsqueeze(0)).squeeze() 
            new_feature_map = self.model.featureMap(new_image.unsqueeze(0)).squeeze()

        # Compute the new one hot encoded vector
        one_hot = torch.zeros(self.prediction.shape, device=self.device)
        one_hot[torch.argmax(new_prediction)] = 1
        self.one_hot = one_hot

        next_state = torch.cat((new_feature_map.type(torch.DoubleTensor), new_prediction.type(torch.DoubleTensor), self.one_hot.type(torch.DoubleTensor)), 0).type(torch.FloatTensor)

        #check if image is misclassified and compute reward 
        episode_done = False
       

        ########################
        ### calcul du reward ### 
        ########################

        # implementation des rewards de l'article 
        w1, w2, w3, w4, w5 = 1, 1, 1, 1, 1
        c = 1

        # pour predicions des autres classes 
        predictions_other_classes = np.zeros(9) 
        predictions_other_classes_previous = np.zeros(9) 
        counter = 0
        for i, elem in enumerate(new_prediction): 
            if(i != self.label):
                predictions_other_classes[counter] = new_prediction[i].to("cpu").numpy()
                predictions_other_classes_previous[counter] = self.prediction[i].to("cpu").numpy()
                counter += 1
                 
        # max prediction qui n'est pas le true label 
        max_pred_other_classes = np.max(predictions_other_classes)

        # get prediction value for original class 
        original_prediction = new_prediction[self.label].to("cpu").numpy()
        original_prediction_previous = self.prediction[self.label].to("cpu").numpy()
        
        # if target_class = -1, it is an untargeted attack 
        if(target_class != -1):
            target_prediction = new_prediction[target_class].to("cpu").numpy()   # we dont do targeted attacks for now 
            target_prediction_previous = self.prediction[target_class].to("cpu").numpy()
        else: 
            target_prediction = predictions_other_classes.max()
            target_prediction_previous = predictions_other_classes_previous.max()
        
        # r1 = w1 * target_prediction 
        r2 = w2 * (target_prediction - target_prediction_previous)
        # r3 = w3 * (target_prediction - original_prediction) 
        # r4 = w4 * (max(0, (target_prediction - max_pred_other_classes)))
        r5 = w5 * torch.norm(perturbation).to("cpu").numpy()
        # r6 = -c

        # fonctions de rewards que l'on test
        reward = - r5 
        # reward = r2

        # fonction de reward du deuxieme article
        # reward = 0
        # if r2 > 0: 
        #     reward += 4
        # else: 
        #     reward = reward - 2
        # if original_prediction - original_prediction_previous < 0: 
        #     reward += 4
        # else: 
        #     reward = reward -2
        # reward = reward - 0.2 * r5

        # check if episode is done 
        if(torch.argmax(new_prediction) != self.label):
            episode_done = True 
            reward = reward + 1000
            # print("real class:", self.label) 
            # print("predicted class:", torch.argmax(new_prediction).to("cpu").numpy())
            # print("prediction vector:", new_prediction.to("cpu").numpy())

        # update prediction and feature map before exiting
        self.prediction = new_prediction
        self.feature_map = new_feature_map 
        self.img = new_image 
        self.perturbation = perturbation
        

        return next_state, reward, episode_done 

    
