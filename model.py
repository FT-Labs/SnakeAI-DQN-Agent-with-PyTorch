#/usr/bin/python

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as fnc
import os


#Inherit from torch nn module
class LinearQNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model_layers = list()
        self.initiliazeLayers(input_size, hidden_size, output_size)


    def initiliazeLayers(self, input_size, hidden_size, output_size):

        for i in range(len(hidden_size)):
            if i == 0:
                layer = nn.Linear(input_size,hidden_size[i])
            elif i != len(hidden_size)-1:
                layer = nn.Linear(hidden_size[i], hidden_size[i+1])
            else:
                layer = nn.Linear(hidden_size[i], output_size)

            self.model_layers.append(layer)

    def forwardPropagation(self, x):

        x = fnc.relu(self.model_layers[0])
        for i in range(len(self.model_layers)-1):
            x = fnc.relu(self.model_layers[i](x))
        x = self.model_layers[i](x)

        return x


    def saveModel(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_name)


class Qtrainer:
    def __init__(self, model, LEARNING_RATE, gamma):
        self.LEARNING_RATE = LEARNING_RATE
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        self.loss_function = nn.MSELoss()
