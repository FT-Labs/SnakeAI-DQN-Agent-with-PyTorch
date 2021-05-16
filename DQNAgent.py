#/usr/bin/python

import torch
import numpy as np
import random
from collections import deque
from model import LinearQNet, Qtrainer
from plotter import plot
import config
import os


# Constant parameters for DQN
MAX_MEMORY = 150_000
BATCH_SIZE = 1500
LEARNING_RATE = 0.001


class DQNAgent:

    def __init__(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers
        self.noOfGames = config.NO_OF_GAMES
        #To controlling randomness of the game
        self.epsilon = 0
        #Discount rate for agent, needs to be smaller than 1
        self.gamma = 0.9
        #Using deque data structure for memory allocation
        #If memory is exceeded, deque will pop front to conserve memory
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, hiddenLayers, 3)
        self.trainer = Qtrainer(self.model, LEARNING_RATE, self.gamma)




    def getState(self, snakeGame):
        head = snakeGame.snake[0]
        pointLeft = config.Loc(head.x - config.BLOCK_SIZE, head.y)
        pointRight = config.Loc(head.x + config.BLOCK_SIZE, head.y)
        pointDown = config.Loc(head.x, head.y + config.BLOCK_SIZE)
        pointUp = config.Loc(head.x, head.y - config.BLOCK_SIZE)

        dirLeft = snakeGame.direction == config.Direction.LEFT
        dirRight = snakeGame.direction == config.Direction.RIGHT
        dirUp = snakeGame.direction == config.Direction.UP
        dirDown = snakeGame.direction == config.Direction.DOWN


        state = [
                (dirRight and snakeGame.onCollisionEnter2D(pointRight)) or
                (dirLeft and snakeGame.onCollisionEnter2D(pointLeft)) or
                (dirUp and snakeGame.onCollisionEnter2D(pointUp)) or
                (dirDown and snakeGame.onCollisionEnter2D(pointDown)),

                #Danger from right
                (dirRight and snakeGame.onCollisionEnter2D(pointDown)) or
                (dirLeft and snakeGame.onCollisionEnter2D(pointUp)) or
                (dirUp and snakeGame.onCollisionEnter2D(pointRight)) or
                (dirDown and snakeGame.onCollisionEnter2D(pointLeft)),

                #Danger from left
                (dirRight and snakeGame.onCollisionEnter2D(pointUp)) or
                (dirLeft and snakeGame.onCollisionEnter2D(pointDown)) or
                (dirUp and snakeGame.onCollisionEnter2D(pointLeft)) or
                (dirDown and snakeGame.onCollisionEnter2D(pointRight)),

                #Move direction
                dirLeft, dirRight, dirUp, dirDown,

                #Food location
                snakeGame.food.x < snakeGame.head.x, #Food left
                snakeGame.food.x > snakeGame.head.x, #Food right
                snakeGame.food.y < snakeGame.head.y, #Food up
                snakeGame.food.y > snakeGame.head.y  #Food down
                ]

        return np.array(state, dtype=int)



    def remember(self, state, action, reward, nextState, gameOver):
        #Store data as a tuple
        self.memory.append((state, action, reward, nextState, gameOver)) # If max memory is reached or exceeded, pop front


    def trainWithLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSampleBatch = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            miniSampleBatch = self.memory

        states, actions, rewards, nextStates, gameOvers = zip(*miniSampleBatch)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)






    def trainWithShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)


    def getAction(self, state):
        #Make random moves: (i.e tradeoff exploration / exploitation)
        #Random moves to explore the current environment and learn
        #After learning, randomness decreases due to knowing which move to choose
        #This can be thinked as generating random parameters in linear regression etc.
        #More games game trains, lower the epsilon, decreasing randomness
        self.epsilon = 100 - self.noOfGames
        finalMove = [0, 0, 0]

        if np.random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            finalMove[move] = 1
        else:
            stateZ = torch.tensor(state, dtype=torch.float)
            prediction = self.model(stateZ)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove




def train(snakeGame):
    pltScores = list()
    pltMeanScores = list()
    totScore = 0
    record = 0
    agent = DQNAgent([256])
    #snakeGame = SnakeGameAI()

    while True:
        #Get previous state
        oldState = agent.getState(snakeGame)

        #Get current move
        finalMove = agent.getAction(oldState)

        #Perform move on game, then get the new state
        reward, gameOver, score = snakeGame.playStep(finalMove)
        newState = agent.getState(snakeGame)


        #Train with short memory
        agent.trainWithShortMemory(oldState, finalMove, reward, newState, gameOver)

        #Remembering the states
        agent.remember(oldState, finalMove, reward, newState, gameOver)

        if gameOver:
            # Replay memory (experience replay) on long memory
            # Plot initial result

            snakeGame.reset()
            agent.noOfGames += 1
            agent.trainWithLongMemory()

            if score > config.RECORD:
                print("MODEL SAVED")
                config.SaveScores(agent.noOfGames, score, agent.model.model_layers)
                agent.model.saveModel()


            print(f"Game: {agent.noOfGames} | Score: {score} | Record: {record}")
            pltScores.append(score)
            totScore += score
            meanScore = totScore / agent.noOfGames
            pltMeanScores.append(meanScore)
            plot(pltScores, pltMeanScores)
