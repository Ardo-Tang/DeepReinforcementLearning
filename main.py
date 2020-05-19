import os, sys
import numpy as np 

import gym

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, ReLU
from keras.models import Model 
import keras.backend as K
import tensorflow as tf

class CarRacing:
    
    def __init__(self, gameName):
        self.env = gym.make(gameName)
        
        self.gamma = 0.99

    def __build_model(self):
        inputLayer = Input(shape=self.env.observation_space.shape, name="input_layer")

        convLayer = Conv2D(8, 3, padding="same", name="conv-1")(inputLayer)
        poolingLayer = MaxPooling2D(3, padding="same", name="pool-1")(convLayer)
        activationLayer = ReLU(max_value=1.0, negative_slope=0.05)(poolingLayer)
        
        convLayer = Conv2D(16, 3, padding="same", name="conv-2")(activationLayer)
        poolingLayer = MaxPooling2D(2, padding="same", name="pool-2")(convLayer)
        activationLayer = ReLU(max_value=1.0, negative_slope=0.05)(poolingLayer)

        convLayer = Conv2D(32, 3, padding="same", name="conv-3")(activationLayer)
        poolingLayer = MaxPooling2D(2, padding="same", name="pool-3")(convLayer)
        activationLayer = ReLU(max_value=1.0, negative_slope=0.05)(poolingLayer)
        
        flattenLayer = Flatten()(activationLayer)

        denseLayer = Dense()

if __name__ == "__main__":
    game = CarRacing("CarRacing-v0")