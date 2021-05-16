#!/usr/bin/python

import pygame
import numpy as np
import config
import sys


pygame.init()
font = pygame.font.Font("./SF-Mono-Regular.otf", 25)
apple = pygame.image.load("./Assets/apple1.png")
appleRect = apple.get_rect()



#RGB Color for blocks
WHITE = (255, 255, 255)
RED = (222, 0 ,0)
GREEN_1 = (0, 255, 0)
GREEN_2 = (0, 0, 0)
BLACK = (0, 0, 0)


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
        self.direction = config.Direction.RIGHT

        self.head = config.Loc(self.width/2, self.height/2)
        self.snake = [self.head, config.Loc(self.head.x - config.BLOCK_SIZE, self.head.y),config.Loc(self.head.x - (2 * config.BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.placeFood()
        self.gameIter = 0




    def placeFood(self):
        x = np.random.randint(0, (self.width - config.BLOCK_SIZE) // config.BLOCK_SIZE) * config.BLOCK_SIZE
        y = np.random.randint(0, (self.height - config.BLOCK_SIZE) // config.BLOCK_SIZE) * config.BLOCK_SIZE

        self.food = config.Loc(x, y)
        if self.food in self.snake:
            self.placeFood()



    def playStep(self, dir=None):
        self.gameIter += 1

        #For game configurations to take action

        for event in pygame.event.get():

                if event.type == pygame.KEYDOWN:
                    if self.humanPlay:
                        if event.key == pygame.K_LEFT:
                            if self.direction != config.Direction.RIGHT:
                                self.direction = config.Direction.LEFT
                        elif event.key == pygame.K_RIGHT:
                            if self.direction != config.Direction.LEFT:
                                self.direction = config.Direction.RIGHT
                        elif event.key == pygame.K_UP:
                            if self.direction != config.Direction.DOWN:
                                self.direction = config.Direction.UP
                        elif event.key == pygame.K_DOWN:
                            if self.direction != config.Direction.UP:
                                self.direction = config.Direction.DOWN
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()



        # Moving the snake (i.e. Updating the head direction of snake)


        if self.humanPlay:
            self.moveSnake(self.direction)
        else:
            self.moveSnake(dir)

        self.snake.insert(0, self.head)


        # Check if the current game is over
        reward = 0
        gameOver = False

        if self.onCollisionEnter2D() or (self.humanPlay == False and self.gameIter > config.STOP_ITER * len(self.snake)):
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
        self.clock.tick(config.SPEED)

        # Return game over (boolean) and current game score
        return reward, gameOver, self.score

    def onCollisionEnter2D(self, loc=None):
        #Check if snake is on the boundary (i.e. wall is hit by snake)

        if loc is None:
            loc = config.Loc(self.head.x, self.head.y)




        if loc.x > self.width - config.BLOCK_SIZE or loc.x < 0 or loc.y > self.height - config.BLOCK_SIZE or loc.y < 0:
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
            pygame.draw.rect(self.display, GREEN_1, pygame.Rect(point.x, point.y, config.BLOCK_SIZE, config.BLOCK_SIZE))

            pygame.draw.rect(self.display, GREEN_2, pygame.Rect(point.x , point.y , config.BLOCK_SIZE, config.BLOCK_SIZE), 3)


        appleRect = apple.get_rect(topleft=(self.food.x, self.food.y))
        #pygame.blit(self.display, RED, pygame.Rect(self.food.x, self.food.y, config.BLOCK_SIZE, config.BLOCK_SIZE))
        self.display.blit(apple, appleRect)

        txt = font.render(f"SCORE: {self.score}", True, WHITE)
        self.display.blit(txt, [0, 0])
        pygame.display.flip()


    def moveSnake(self, dir):

        clockWise = [config.Direction.RIGHT, config.Direction.DOWN, config.Direction.LEFT, config.Direction.UP]
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


        if self.direction == config.Direction.RIGHT:
            x += config.BLOCK_SIZE
        elif self.direction == config.Direction.LEFT:
            x -= config.BLOCK_SIZE
        elif self.direction == config.Direction.DOWN:
            y += config.BLOCK_SIZE
        elif self.direction == config.Direction.UP:
            y -= config.BLOCK_SIZE

        self.head = config.Loc(x, y)


    def humanGame(self):
        self.humanPlay = True

        #Rendering game
        while True:
            reward, gameOver, score = self.playStep()

            if gameOver == True:
                break

        print(f"Final Score: {score}")

        pygame.quit()
