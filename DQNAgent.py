#/usr/bin/python

import torch
import numpy as np
from collections import deque
from snakeClass import SnakeGameAI, Direction, Loc, BLOCK_SIZE


# Constant parameters for DQN
MAX_MEMORY = 150_000
BATCH_SIZE = 1500
LEARNING_RATE = 0.001


class DQNAgent:

    def __init__(self):
        self.noOfGames = 0
        #To controlling randomness of the game
        self.epsilon = 0
        #Discount rate for agent
        self.gamma = 0
        #Using deque data structure for memory allocation
        #If memory is exceeded, deque will pop front to conserve memory
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None
        self.trainer = None




    def getState(self, snakeGame):
        head = snakeGame.head[0]
        pointLeft = Loc(head.x - BLOCK_SIZE, head.y)
        pointRight = Loc(head.x + BLOCK_SIZE, head.y)
        pointDown = Loc(head.x, head.y + BLOCK_SIZE)
        pointUp = Loc(head.x, head.y - BLOCK_SIZE)

        dirLeft = snakeGame.direction == Direction.LEFT
        dirRight = snakeGame.direction == Direction.RIGHT
        dirUp = snakeGame.direction == Direction.UP
        dirDown = snakeGame.direction == Direction.DOWN


        state = [
                (dirRight and snakeGame.onCollisionEnter2D(pointRight)) or
                (dirLeft and snakeGame.onCollisionEnter2D(pointLeft)) or
                (dirUp and snakeGame.onCollisionEnter2D(pointUp)) or
                (dirDown and snakeGame.onCollisionEnter2D(pointDown)),

                #Danger from right
                (dirRight and snakeGame.onCollisionEnter2D(dirDown)) or
                (dirLeft and snakeGame.onCollisionEnter2D(dirUp)) or
                (dirUp and snakeGame.onCollisionEnter2D(dirRight)) or
                (dirDown and snakeGame.onCollisionEnter2D(dirLeft)),

                #Danger from left
                (dirRight and snakeGame.onCollisionEnter2D(dirUp)) or
                (dirLeft and snakeGame.onCollisionEnter2D(dirDown)) or
                (dirUp and snakeGame.onCollisionEnter2D(dirLeft)) or
                (dirDown and snakeGame.onCollisionEnter2D(dirRight)),

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
            miniSampleBatch = np.random.choice(self.memory, BATCH_SIZE) # List of tuples
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
            move = np.random.randint(0, 3, size=1)
            finalMove[move[0]] = 1
        else:
            stateZ = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(stateZ)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove




def train():
    pltScores = list()
    pltMeanScores = list()
    totScore = 0
    record = 0
    agent = DQNAgent()
    snakeGame = SnakeGameAI()

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

            if score > record:
                record = score
                # TODO // agent.model.save()

            print(f"Game: {agent.noOfGames} | Score: {score} | Record: {record}")

            #TODO // plot













if __name__ == "__main__":
    train()
