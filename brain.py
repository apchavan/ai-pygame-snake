#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers


class Brain:
    def __init__(self, input_shape=(100, 100, 4), learning_rate=0.0005):
        self.inputShape = input_shape
        self.learningRate = learning_rate
        self.numOutputs = 4

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=self.inputShape))
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation="relu"))
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(units=256, activation="relu"))
        self.model.add(layers.Dense(units=self.numOutputs))

        self.model.compile(loss="mean_squared_error", optimizer=optimizers.Adam(lr=self.learningRate))

    def loadModel(self, filename: str):
        self.model = models.load_model(filename)
        return self.model

