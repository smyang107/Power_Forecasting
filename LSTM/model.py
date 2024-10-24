import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn import metrics
from tensorflow.keras.layers import Dense, LSTM, Bidirectional,GRU
import csv
from tensorflow.keras.models import Sequential

from tensorflow import keras

class Mymodel(keras.Model):
    def __init__(self, uint1, uint2, input_shape, **kwargs):
        super(Mymodel, self).__init__(**kwargs)
        # self.model = Sequential()
        # self.model.add(GRU(units=uint1, activation='tanh', return_sequences=True,input_shape=(input_shape, 1)))
        # self.model.add(GRU(units=uint2, activation='tanh'))
        # self.model.add(Dense(10))
        # self.model.add(Dense(1))
        # self.summary = self.model.summary()
        self.layer1 = GRU(units=uint1, activation='tanh', return_sequences=True, input_shape=(input_shape, 1))
        self.layer2 = GRU(units=uint2, activation='tanh')
        self.layer2 = GRU(units=uint2, activation='tanh')
        self.fc = Dense(10)
        self.outLayer = Dense(1)

    def call(self,inputs,training=None):

        # out = self.model(inputs,training=training) #无残差连接
        # return out #残差连接
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.fc(x)
        out = self.outLayer(x)
        return out
