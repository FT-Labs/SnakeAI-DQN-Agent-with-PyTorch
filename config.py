#/usr/bin/python

from enum import Enum
from collections import namedtuple
import json
import os

BLOCK_SIZE = 32
SPEED = 16
STOP_ITER = 100
RECORD = 0
NO_OF_GAMES = 0

def LoadScores():
    global NO_OF_GAMES
    global RECORD

    if not os.path.isfile('./model/model.pth'):
        SaveScores(0, 0)

    try:
        f = open("scores.json", 'r')
        data = json.load(f)
        NO_OF_GAMES = data['numberOfGames']
        RECORD = data['record']
        f.close()
    except Exception as e:
        print(e)
        f = open("scores.json", 'w')
        dct = {
                "numberOfGames" : 0,
                "record" : 0
                }
        json.dump(dct, f)
        f.close()

def SaveScores(numberOfGames, record, modelLayers=list()):
    global NO_OF_GAMES
    global RECORD
    NO_OF_GAMES = numberOfGames
    RECORD = record
    MODEL_LAYERS = "\n".join([str(model) for model in modelLayers])

    dct = {
            "numberOfGames" : NO_OF_GAMES,
            "record" : RECORD,
            "layers" : MODEL_LAYERS
            }

    f = open("scores.json", 'w')
    json.dump(dct, f)
    f.close()

    print(f"Model with score: {RECORD} and number of games with {NO_OF_GAMES} saved succesfully.")


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Loc = namedtuple("location", ['x', 'y'])
