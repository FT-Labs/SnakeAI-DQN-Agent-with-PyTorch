import distutils
from distutils import util
import sys
import argparse
import snakeClass
import DQNAgent
import config



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--humanplay", nargs='?', type=distutils.util.strtobool, default=False)
    parser.add_argument("--speed", nargs='?', type=int, default=config.SPEED)
#    parser.add_argument("--help", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()
    config.SPEED = args.speed

    game = snakeClass.SnakeGameAI()

    if args.humanplay:
        game.humanGame()
    else:
        DQNAgent.train(game)
