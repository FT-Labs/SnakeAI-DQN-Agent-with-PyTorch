#/usr/bin/python

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as fnc
import os
import sys


#Inherit from torch nn module
class LinearQNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model_layers = nn.ModuleList()
        self.initiliazeLayers(input_size, hidden_size, output_size)
        self.loadModel()



    def initiliazeLayers(self, input_size, hidden_size, output_size):

        for i in range(len(hidden_size)+1):
            if i == 0:
                layer = nn.Linear(input_size,hidden_size[i])
            elif i != len(hidden_size):
                layer = nn.Linear(hidden_size[i-1], hidden_size[i])
            else:
                layer = nn.Linear(hidden_size[i-1], output_size)

            self.model_layers.append(layer)


    def forward(self, x):

        for i in range(len(self.model_layers)-1):
            x = fnc.relu(self.model_layers[i](x))
        x = self.model_layers[-1](x)

        return x


    def saveModel(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


    def loadModel(self, file_name='model.pth'):
        model_folder_path = "./model"
        try :
                self.load_state_dict(torch.load(os.path.join(model_folder_path, file_name)), strict=False)
                self.eval()
                print("Model loaded succesfully")
        except Exception as e:
            print(e)


class Qtrainer:
    def __init__(self, model, LEARNING_RATE, gamma):
        self.LEARNING_RATE = LEARNING_RATE
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.loss_function = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, gameOver):

        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # If length is one, need to transform it into a 2d array (1, x)
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            # Code above appends 1 dimension to x axis
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # Since gameOver is a boolean with one value, it needs to be converted to a tuple
            gameOver = (gameOver, )

        # Get predicted Q values with current state
        pred = self.model(state)

        # Starting Bellman Equation : r + y * max(next predicted Q value)
        target = pred.clone()
        for i in range(len(gameOver)):
            newQ = reward[i]
            if not gameOver[i]:
                newQ = reward[i] + self.gamma * torch.max(self.model(nextState[i]))


            m = torch.argmax(action[i]).item()
            target[i][m] = newQ

        self.optimizer.zero_grad() # Empty the gradient in pytorch, this needs to remembered
        loss = self.loss_function(target, pred)

        # Execute backward prop
        loss.backward()

        self.optimizer.step()
