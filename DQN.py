#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


class Dqn:
    def __init__(self, max_memory: tf.uint32=100, discount_factor: tf.float32=0.9):
        self.xpMemory: list = list()
        self.maxMemory: tf.uint32 = max_memory
        self.discountFactor: tf.float32 = discount_factor
        self.randomGenerator = np.random.default_rng()

    def remember(self, transition, game_over):
        self.xpMemory.append([transition, game_over])
        if len(self.xpMemory) > self.maxMemory:
            del self.xpMemory[:1]

    def get_batch(self, model, batch_size: tf.uint32=10):
        xpMemoryLength: int = len(self.xpMemory)
        inputs = np.zeros((min(batch_size, xpMemoryLength), self.xpMemory[0][0][0].shape[1],
                           self.xpMemory[0][0][0].shape[2], self.xpMemory[0][0][0].shape[3]))
        targets = np.zeros((min(batch_size, xpMemoryLength), model.output_shape[-1]))

        for i, idx in enumerate(self.randomGenerator.integers(0, xpMemoryLength, size=min(xpMemoryLength, batch_size))):
            current_state, action, reward, next_state = self.xpMemory[idx][0]
            game_over = self.xpMemory[idx][1]

            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            if game_over:
                print(f"\t\t IF =>> i = {i}, action = {action} ")
                targets[i, action] = reward
            else:
                print(f"\t\t ELSE =>> i = {i}, action = {action} ")
                targets[i, action] = reward + self.discountFactor * Q_sa
        return inputs, targets

