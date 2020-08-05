#!/usr/bin/env python3

from environment import Environment
from brain import Brain

import numpy as np
import tensorflow as tf


# Defining the parameters
filePathToOpen: str = "model.h5"
nLastStates: int = 4
slowdown: int = 75

env = Environment(slowdown)
brain = Brain(input_shape=(env.nRows, env.nColumns, nLastStates))
model = brain.loadModel(filePathToOpen)


# Making a function that will reset game states
def resetStates():
    currentState = np.zeros((1, env.nRows, env.nColumns, nLastStates))
    for i in range(nLastStates):
        currentState[:, :, :, i] = env.screenMap
    return currentState, currentState



# Starting the main loop
while True:
    env.reset()
    currentState, nextState = resetStates()
    gameOver = False

    while not gameOver:

        q_values = model.predict(currentState)[0]
        action = np.argmax(q_values)

        state, _, gameOver = env.step(action=action)
        state = np.reshape(state, (1, env.nRows, env.nColumns, 1))
        nextState = np.append(nextState, state, axis=3)
        nextState = np.delete(nextState, 0, axis=3)

        currentState = nextState
