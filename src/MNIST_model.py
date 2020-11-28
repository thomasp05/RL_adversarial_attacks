# this is where we implement the state-of-the-art MNIST model that will be under attack 


import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms 
import numpy as np 
import time


class Lenet5(nn.Module): 
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(1, 6, 5) 
        self.S2 = nn.AvgPool2d(2, 2)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5) 
        self.S4 = nn.AvgPool2d(2) 
        self.F5 = nn.Linear(16*4*4, 120)
        self.F6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10) 

    def forward(self, x): 
        y = self.C1(x)
        y = F.relu(y)
        y = self.S2(y)

        y = self.C3(y)
        y = F.relu(y)
        y = self.S4(y)
     
        y = y.view(-1, 16*4*4)  
        y = self.F5(y)
        y = F.relu(y)

        y = self.F6(y)
        y = F.relu(y)

        y = self.output(y)

        return y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# define the tranformation for the data 
transform = transforms.Compose([transforms.ToTensor(),
                                ])

# load the training and validation datasets 
trainset = datasets.MNIST('data/mnist/trainset', train = True, download=True, transform=transform)
valset = datasets.MNIST('data/mnist/validation', train=False, download=True, transform=transform)

#load the data to the dataloader 
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)
valloader = torch.utils.data.DataLoader(valset, shuffle=True,  batch_size=64)

# train the model 
nb_epoch = 12
batch_size = 64 
learning_rate = 0.01
momentum = 0.9

print("CUDA available: ", torch.cuda.is_available())
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 
model = Net().to(device) 
# model.train()   # set the model in training mode 

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = torch.nn.CrossEntropyLoss()

time0 = time.time()
for i_epoch in range(nb_epoch): 
    running_loss = 0
    for i_batch, (images, labels) in enumerate(trainloader): 
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                                # it is required to reset the gradient values to 0 
        output = model(images)                               # compute the prediction 
        loss = criterion(output, labels)                     # compute the error 
        loss.backward()                                      # derive the graph (for the gradients)
        optimizer.step()                                     # take an optimization step to update the parameters of the model 
        running_loss += loss.item()                         
    else: 
        print("Epoch {} - training loss: {}".format(i_epoch, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =", (time.time()-time0)/60)

# finally, compute the error in generalization 
model.eval()     # now the model is in testing mode 

correct_count = 0 
all_count = 0 
for images, labels in valloader: 
    images, labels = images, labels
    for i in range(len(labels)): 
        image = images[i].view(1, 1, 28, 28).to(device)
        label = labels.numpy()[i]
        with(torch.no_grad()):
            output = list(model(image).to("cpu").numpy()[0])
         
            prediction = output.index(max(output))
        
            if(prediction == label): 
                correct_count += 1
            all_count += 1

print("Number of Images Tested=", all_count) 
print("\nModel Accuracy =", (correct_count/all_count)*100)

        
# save the model 
state = model.state_dict()
torch.save(state, 'models/mnist_lenet.pt')

