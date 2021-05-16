#/usr/bin/python

from enum import Enum
from collections import namedtuple

BLOCK_SIZE = 32
SPEED = 16
STOP_ITER = 100

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Loc = namedtuple("location", ['x', 'y'])
