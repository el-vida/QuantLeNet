import matplotlib.pyplot as plt

import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

import torch.optim as optim
from torch.utils.data import DataLoader

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


# GPU availability check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# train and test data generation, using MNIST dataset, converting the images into tensors
train_data = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True )
test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor())
train_data , val_data = random_split(train_data,[50000,10000])


# Dataloaders for shuffling, batch-sizing of data set during training and testing
batch_size= 128
trainloader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
valloader = DataLoader(val_data,batch_size=batch_size,shuffle=False)
testloader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

classes = ('0','1','2','3','4','5','6','7','8','9')

# Defining a custom LeNet module
# This is a LeNet architecture which consist of two convolution layer , 
# two pooling layers , three fully connected layers 
# and tanh activation function attached after every layer .
class LeNet(nn.Module):                         # Extending nn.Module class 
    def __init__(self):                         # Constructor 
        super(LeNet,self).__init__()            # Calls the constructor of nn.Module
        self.cnn_model = nn.Sequential(         # nn.Sequential allows multiple layers to stack together
            nn.Conv2d(1,6,5),                   #(N,1,28,28) -> (N,6,24,24)
            nn.Tanh(),                      
            nn.AvgPool2d(2,stride=2),           #(N,6,24,24) -> (N,6,12,12)
            nn.Conv2d(6,16,5),                  #(N,6,12,12) -> (N,16,8,8)
            nn.Tanh(),
            nn.AvgPool2d(2,stride=2)            #(N,16,8,8) -> (N,16,4,4)
            )
        
        self.fc_model = nn.Sequential(          # Fully connected layer 
            nn.Linear(256,120),
            nn.Tanh(),
            nn.Linear(120,84),
            nn.Tanh(),
            nn.Linear(84,10)
        
            )
        
# It gets a batch of data which we have defined earlier 
        
    def forward(self,x):     
        #print(x.shape)
        x = self.cnn_model(x)       
        #print(x.shape)
        x = x.view(x.size(0),-1)    # Flatening the inputs from tensors to vectors 
        #print(x.shape)
        x = self.fc_model(x)        # Passing the conv layer to fully connected layer
        #print(x.shape)
        return x


# Quantization using BREVITAS
class QuantWeightActLeNet(nn.Module):
    def __init__(self):
        super(QuantWeightActLeNet, self).__init__() #LowPrecisionLeNet
        self.quant_inp = qnn.QuantIdentity(bit_width=4) # for very first iteration
        self.cnn_model = nn.Sequential(         # nn.Sequential allows multiple layers to stack together
            qnn.QuantConv2d(1,6,5, bias=True, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4),                      
            qnn.QuantAvgPool2d(2,stride=2,trunc_quant=None, return_quant_tensor=True),           #(N,6,24,24) -> (N,6,12,12)
            qnn.QuantConv2d(6,16,5, bias=True, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantAvgPool2d(2,stride=2,trunc_quant=None, return_quant_tensor=True),
            )
        
        self.fc_model = nn.Sequential(
            qnn.QuantLinear(256,120, bias=True, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(120,84, bias=True, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(84,10, bias=True, weight_bit_width=4)
        )
    
        # self.cnn_model = nn.Sequential(
        #     qnn.QuantConv2d(1,6,5, bias=True, weight_bit_width=4),
        #     #qnn.QuantTanh(bit_width=4),
        #     nn.Tanh(),
        #     qnn.QuantAvgPool2d(2, stride = 2),
        #     qnn.QuantConv2d(6,16,5, bias=True, weight_bit_width=4),
        #     #qnn.QuantTanh(bit_width=4),
        #     nn.Tanh(),
        #     qnn.QuantAvgPool2d(2,stride=2)
        # )

        # self.fc_model = nn.Sequential(
        #     qnn.QuantLinear(256,120, bias=True, weight_bit_width=4),
        #     #qnn.QuantTanh(bit_width=4),
        #     nn.Tanh(),
        #     qnn.QuantLinear(120,84, bias=True, weight_bit_width=4),
        #     #qnn.QuantTanh(bit_width=4),
        #     nn.Tanh(),
        #     qnn.QuantLinear(84,10, bias=True, weight_bit_width=4)
        # )

# It gets a batch of data which we have defined earlier     
    def forward(self,x):     
        #print(x.shape)
        x = self.quant_inp(x)
        x = self.cnn_model(x)       
        #print(x.shape)
        x = x.view(x.size(0),-1)    # Flatening the inputs from tensors to vectors 
        #print(x.shape)
        x = self.fc_model(x)        # Passing the conv layer to fully connected layer
        #print(x.shape)
        return x
    

# # Evaluate our model
# def evaluation(dataloader):
#     total , correct = 0,0
#     for data in dataloader:
#         inputs , labels = data
#         inputs , labels = inputs.to(device) , labels.to(device)
#         output = net(inputs)            
#         max_pred,pred = torch.max(output.data,dim=1)
#         total +=labels.size(0)
#         correct +=(pred == labels).sum().item()  
#     return 100 * correct / total

# Evaluate our model
def evaluation_new(dataloader, net):
    total , correct = 0,0
    for data in dataloader:
        inputs , labels = data
        inputs , labels = inputs.to(device) , labels.to(device)
        output = net(inputs)            
        max_pred,pred = torch.max(output.data,dim=1)
        total +=labels.size(0)
        correct +=(pred == labels).sum().item()  
    return 100 * correct / total


# def fit(max_epochs =2):
    
#     loss_arr = []
#     loss_epoch_arr = []
    
#     for epoch in range(max_epochs):
#         for i, data in enumerate(trainloader,0): # Iterating through the train loader 
#             inputs,labels = data
#             inputs,labels = inputs.to(device),labels.to(device)

#             opt.zero_grad()     # Reset the gradient in every iteration

#             outputs = net(inputs)
#             loss = loss_fn(outputs,labels)   # Loss forward pass
#             loss.backward()                  # Loss backend pass
#             opt.step()                       # Update all the parameters by the given learnig rule

#             loss_arr.append(loss.item())
#         loss_epoch_arr.append(loss.item())
#         print('Val accuracy: %0.2f , Train accuracy : %0.2f'%(evaluation(valloader),evaluation(trainloader)))

#     plt.plot(loss_epoch_arr)
#     plt.show(block=True)

def fit2(max_epochs =16):

    net1 = LeNet().to(device)           # Creating object for LeNet() model and passing it to GPU 

    loss_fn1 = nn.CrossEntropyLoss()    # It takes the highest value which is the predictions and mark it as 1
                                      # And mark rest of the values as zeros. 

    opt1 = optim.Adam(net1.parameters()) # Using adam Optimization algorithm , we can also specify the hyperparameters .

    
    loss_arr1 = []
    loss_epoch_arr1 = []

    net2 = QuantWeightActLeNet().to(device)           # Creating object for QuantLeNet() model and passing it to GPU 

    loss_fn2 = nn.CrossEntropyLoss()    # It takes the highest value which is the predictions and mark it as 1
                                    # And mark rest of the values as zeros. 

    opt2 = optim.Adam(net2.parameters()) # Using adam Optimization algorithm , we can also specify the hyperparameters .

    loss_arr2 = []
    loss_epoch_arr2 = []
    
    for epoch in range(max_epochs):
        for i, data in enumerate(trainloader,0): # Iterating through the train loader 
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)

            opt1.zero_grad()     # Reset the gradient in every iteration
            opt2.zero_grad()

            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            loss1 = loss_fn1(outputs1,labels)   # Loss forward pass
            loss2 = loss_fn2(outputs2, labels)
            loss1.backward()                  # Loss backend pass
            loss2.backward()
            opt1.step()                       # Update all the parameters by the given learnig rule
            opt2.step()

            loss_arr1.append(loss1.item())
            loss_arr2.append(loss2.item())
        loss_epoch_arr1.append(loss1.item())
        loss_epoch_arr2.append(loss2.item())
        print('LeNet: Val accuracy: %0.2f , Train accuracy : %0.2f'%(evaluation_new(valloader, net1),evaluation_new(trainloader, net1)))
        print('QuantLeNet: Val accuracy: %0.2f , Train accuracy : %0.2f'%(evaluation_new(valloader, net2),evaluation_new(trainloader, net2)))

    plt.plot(loss_epoch_arr1, label="LeNet")
    plt.plot(loss_epoch_arr2, label="QuantLeNet")
    plt.title("LeNet without quantization vs quantized LeNet")
    plt.legend(['LeNet','QuantLeNet'])
    plt.show(block=True)


# print("1: LeNet") # CASE 1: LeNet ----------------------------------------------------------------------------------------------------

# net = LeNet().to(device)           # Creating object for LeNet() model and passing it to GPU 

# loss_fn = nn.CrossEntropyLoss()    # It takes the highest value which is the predictions and mark it as 1
#                                    # And mark rest of the values as zeros. 

# opt = optim.Adam(net.parameters()) # Using adam Optimization algorithm , we can also specify the hyperparameters .
    
# fit()


# print("2: QuantLeNet") # CASE 2: QANT LeNet ----------------------------------------------------------------------------------------------------

# net = QuantWeightActLeNet().to(device)           # Creating object for QuantLeNet() model and passing it to GPU 

# loss_fn = nn.CrossEntropyLoss()    # It takes the highest value which is the predictions and mark it as 1
#                                    # And mark rest of the values as zeros. 

# opt = optim.Adam(net.parameters()) # Using adam Optimization algorithm , we can also specify the hyperparameters .
    
# fit()

fit2()

# SOURCES:
# https://medium.com/analytics-vidhya/train-mnist-dataset-on-lenet-model-6180917c85b6
# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

# REMARKS:
# changed tanh activation to ReLU because brevitas with tanh keeps creating assertion errors --> leads to slightly reduced accuracy values at the beginning
# avgpooling stopped raising an assertion error once 'trunc_quant=None' was specified