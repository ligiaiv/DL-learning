from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint 
from keras.layers import Input, LSTM, GRU, Bidirectional
from keras.models import Model

T = 8 # Segquence lenth
D = 2 #imput dimentionality (embedding)
M = 3 #latent dimentionality

X = np.random.randn(1,T,D)

input_ = Input(shape = (T,D)) #criei a camada de input
rnn = Bidirectional(LSTM(M,return_state = True, return_sequences = False)) # criei a LSTM bidirecional

x = rnn(input_) #disse que o x é a Bidirecional que recebe na entrada a camada input_

#criei um Model que inclui a camada input e a camada rnn
#poderia tambem fazer model = Model()    model,add(x)
model = Model(input=input_,outputs=x)
o,h1,c1,h2,c2 = model.predict(X)

print("o: ",o)
print("o.shape: ",o.shape)
print("h1: ",h1)
print("c1: ",c1)
print("h2: ",h2)
print("c2: ",c2)


