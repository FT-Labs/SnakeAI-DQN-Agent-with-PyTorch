#!/usr/bin/python

import pygame
import distutils
from distutils import util
import sys
import argparse
import numpy as np
from enum import Enum
from collections import namedtuple


pygame.init()
font = pygame.font.Font("./SF-Mono-Regular.otf", 25)
apple = pygame.image.load("./Assets/apple1.png")
appleRect = apple.get_rect()


class Direction(Enum):
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4


Loc = namedtuple("location", "x, y")

#RGB Color for blocks
WHITE = (255, 255, 255)
RED = (222, 0 ,0)
GREEN_1 = (0, 255, 0)
GREEN_2 = (0, 0, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 32
SPEED = 16
STOP_ITER = 100

class SnakeGameAI:

    def __init__(self, width=1024, height=768, humanPlay=False):
        self.width = width
        self.height = height
        self.humanPlay = humanPlay
        self.gameIter = 0

        #Display config
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):

        #Initiliazing game state
        self.direction = Direction.RIGHT

        self.head = Loc(self.width/2, self.height/2)
        self.snake = [self.head, Loc(self.head.x - BLOCK_SIZE, self.head.y),Loc(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.placeFood()
        self.gameIter = 0




    def placeFood(self):
        x = np.random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = np.random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Loc(x, y)
        if self.food in self.snake:
            self.placeFood()



    def playStep(self, dir=None):
        self.gameIter += 1

        #For game configurations to take action

        if self.humanPlay:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        if self.direction != Direction.RIGHT:
                            self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        if self.direction != Direction.LEFT:
                            self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        if self.direction != Direction.DOWN:
                            self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        if self.direction != Direction.UP:
                            self.direction = Direction.DOWN


        # Moving the snake (i.e. Updating the head direction of snake)


        if self.humanPlay:
            self.moveSnake(self.direction)
        else:
            self.moveSnake(dir)

        self.snake.insert(0, self.head)


        # Check if the current game is over
        reward = 0
        gameOver = False

        if self.onCollisionEnter2D() or (self.humanPlay == False and self.gameIter > STOP_ITER * len(self.snake)):
            gameOver = True
            reward = -10
            return reward, gameOver, self.score

        # If game is not over, place a new food or move
        if self.head == self.food:
            self.score += 1
            reward += 10
            self.placeFood()
        else:
            self.snake.pop()


        # Update ui (visuals) and game clock

        self.updateUi()
        self.clock.tick(SPEED)

        # Return game over (boolean) and current game score
        return reward, gameOver, self.score

    def onCollisionEnter2D(self, loc=None):
        #Check if snake is on the boundary (i.e. wall is hit by snake)

        if loc is None:
            loc = self.head




        if loc.x > self.width - BLOCK_SIZE or loc.x < 0 or loc.y > self.height - BLOCK_SIZE or loc.y < 0:
            return True

        #If snake hits itself
        if loc in self.snake[1:]:
            return True

        #If any of the conditions are not satisfied, return false
        return False

    def updateUi(self):

        # Fill the background with black color
        self.display.fill(BLACK)


        for point in self.snake:
            pygame.draw.rect(self.display, GREEN_1, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))

            pygame.draw.rect(self.display, GREEN_2, pygame.Rect(point.x , point.y , BLOCK_SIZE, BLOCK_SIZE), 3)

            appleRect = apple.get_rect(topleft=(self.food.x, self.food.y))
            #pygame.blit(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
            self.display.blit(apple, appleRect)

            txt = font.render(f"SCORE: {self.score}", True, WHITE)
            self.display.blit(txt, [0, 0])
            pygame.display.flip()



    def moveSnake(self, dir):

        clockWise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockWise.index(self.direction)


        if not self.humanPlay:
            if np.array_equal(dir, [1, 0, 0]):
                # No changes
                newDir = clockWise[idx]
            elif np.array_equal(dir, [0, 1, 0]):
                nextIdx = (idx + 1) % 4
                # Take a right turn
                newDir = clockWise[nextIdx]
            else: #i.e [0, 0, 1]
                nextIdx = (idx - 1) % 4
                newDir = clockWise[nextIdx]

            self.direction = newDir



        x = self.head.x
        y = self.head.y


        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Loc(x, y)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--humanplay", nargs='?', type=distutils.util.strtobool, default=False)
    parser.add_argument("--speed", nargs='?', type=int, default=BLOCK_SIZE)
#    parser.add_argument("--help", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()


    snakeGame = SnakeGameAI(humanPlay=args.humanplay)

    #Rendering game
    while True:
        reward, gameOver, score = snakeGame.playStep()

        if gameOver == True:
            break

    print(f"Final Score: {score}")

    pygame.quit()


def displayOptions():
    print("""
Options:
    --humanplay, default=true
    --speed, default={}
""".format(BLOCK_SIZE))
