from __future__ import print_function, division
from builtins import range,input

from keras.models import Model
from keras.layers import Input , LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

T = 8
D = 2
M = 3

X = np.random.randn(1,T,D)

def lstm1():
    input_ = Input(shape = (T,D))
    rnn = LSTM(M, return_state = True)
    x = rnn(input_)

    model = Model(inputs = input_,outputs = x)
    o,h,c = model.predict(X)
    print("o: ",o)
    print("h: ",h)
    print("c: ",c)

def lstm2():
    input_ = Input(shape = (T,D))
    rnn = LSTM(M, return_state = True,return_sequences = True)
    x = rnn(input_)

    model = Model(inputs = input_,outputs = x)
    o,h,c = model.predict(X)
    print("o: ",o)
    print("h: ",h)
    print("c: ",c)

print("lstm1: ")
lstm1()


print("lstm2: ")
lstm2()

