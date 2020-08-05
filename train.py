#!/usr/bin/env python3

from environment import Environment
from brain import Brain
from DQN import Dqn

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Defining the parameters
memSize: tf.uint32 = 60_000
batchSize: tf.uint32 = 32
learningRate: tf.float32 = 0.0001
gamma: tf.float32 = 0.9
nLastStates: int = 4

epsilon: float = 1.0
epsilonDecayRate: float = 0.0002
minEpsilon: float = 0.05
randomGenerator = np.random.default_rng()

filepathToSave: str = "model.h5"

# Creating the Environment, the Brain and the Experience Replay Memory
env = Environment(0)
brain = Brain(input_shape=(env.nRows, env.nColumns, nLastStates), learning_rate=learningRate)
model = brain.model
dqn = Dqn(max_memory=memSize, discount_factor=gamma)

# Making a function that will initialize game states
def resetStates():
    currentState = np.zeros((1, env.nRows, env.nColumns, nLastStates))

    for i in range(nLastStates):
        currentState[:, :, :, i] = env.screenMap

    # Return two 'currentState'; one to represent board before taking action & another after taking action
    return currentState, currentState

# Starting main loop
epoch: tf.uint32 = 0
scores: list = list()            # To save the average scores per game after every 100 games/epochs
maxNCollected: tf.uint32 = 0     # The highest score obtained so far in the training
nCollected: tf.uint32 = 0        # The score in each game/epoch
totalNCollected: tf.uint32 = 0   # How many apples you've collected over 100 epochs/games


while True:
    env.reset()                                # Initially reset the environment
    currentState, nextState = resetStates()    # Initially reset all the states
    epoch += 1
    gameOver = False

    # Starting the second loop in which we play the game and teach our AI
    while not gameOver:
        if randomGenerator.random() < epsilon:
            action = randomGenerator.integers(0, 4)
        else:
            q_values = model.predict(currentState)[0]
            action = np.argmax(q_values)

        # Updating the environment
        state, reward, gameOver = env.step(action=action)
        # Adding new game frame to the next state and deleting the oldest frame from next state
        state = np.reshape(state, (1, env.nRows, env.nColumns, 1))
        nextState = np.append(nextState, state, axis=3)
        nextState = np.delete(nextState, 0, axis=3)

        # Remembering the transition and training our AI
        dqn.remember([currentState, action, reward, nextState], gameOver)

        inputs, targets = dqn.get_batch(model=model, batch_size=batchSize)
        model.train_on_batch(inputs, targets)

        # Checking whether we have collected an apple and updating the current state
        if env.collected:
            nCollected += 1
        currentState = nextState

    # Checking if a record of apples eaten in a round was beaten and if yes then saving the model
    if nCollected > maxNCollected and nCollected > 2.0:
        maxNCollected = nCollected
        model.save(filepathToSave)
    totalNCollected += nCollected
    nCollected = 0

    # Showing the results each 100 games
    if epoch % 100 == 0 and epoch != 0:
        scores.append(totalNCollected / 100)
        totalNCollected = 0
        plt.plot(scores)
        plt.xlabel("Epoch / 100")
        plt.ylabel("Average Score")
        plt.savefig("stats.png")
        plt.close()

    # Lowering the epsilon
    if epsilon > minEpsilon:
        epsilon -= epsilonDecayRate
    # Showing the results each game
    print("Epoch : " + str(epoch) + " Current Best : " + str(maxNCollected) + " Epsilon : {:.5f}".format(epsilon))


