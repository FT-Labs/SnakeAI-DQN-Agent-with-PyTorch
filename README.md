## Deep Q Neural Network with PyTorch Learning to Play Snake 🐍

Snake game has been made with PyTorch. Main class and parameters are in "SnakeGame.py". Below illustrates a trained dqn model, which you can see it by yourself by cloning this repo and follow the instructions below.

---


![alt text](https://github.com/ardaPhysTech/SnakeAI-DQN-Agent-With-PyTorch/blob/feature/snakeai/gif/snakeai.gif "Snake AI")
---
#### Global parameters
Since some of the classes use the same parameters, like BLOCK_SIZE (size of each block of the snake) and SPEED (game speed), these global variables have been written in a config file "config.py".

---

#### Layer sizes
The game uses linear neural network, also the layers can be changed in "DQNAgent.py" in the global "train" function. Feel free to play with layers, see if there is better one then the current one.

---

### How to start the game or train it:
To start the game without parameters, just run the SnakeGame.py file. \
Note that every argument needs to be appended after
```
python SnakeClass.py --parameter 1 --parameter 2
```
Also, to get the parameters of the class:
```
python SnakeClass.py -h
```
##### Parameters:
* \-\-humanplay (boolean,default=False)
* \-\-speed (int, default=16)


For example, if you want to play the game yourself just write: \
	```
	python SnakeClass.py --humanplay true
	```

If no parameters are given, game will start to train itself. If training speed is small, change it with writing:
```
python SnakeClass.py --speed (value)
```

### IMPORTANT !
If game does not run on your computer or give module errors, just pip install all of these modules:
* matplotlib
* torch
* torchvision
* pandas
* json
* pygame
* ipython

E.g: ``` pip install matplotlib pandas ... ```
