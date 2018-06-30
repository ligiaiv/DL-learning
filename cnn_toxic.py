from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.models import Model
from sklearn.metrics import roc_auc_score



MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 100


print('Loading word vectors...')
word2vec = {}
here= os.path.dirname(os.path.realpath(__file__))+('/BigFiles/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)
print(here)
with open(here) as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype = 'float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

print('Loading in comments...')
train = pd.read_csv('../BigFiles/toxic_comments/train.csv')
sentences = train['comment_text'].fillna("DUMMY_VALUE").values
possible_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hater"]
targets = train[possible_labels].values

print("max sequence length",max(len(s) for s in sentences))
print("min sequence length",min(len(s) for s in sentences))
s = sorted(len(s) for s in sentences)
print("median sequence length",s[len(s)//2])

# convert the sentences(strings) into integers

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences) #gives each word a number
sequences = tokenizer.texts_to_sequences(sentences) #replaces each word with its index

word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

data = pad_sequences(sequences,maxlen(MAX_SEQUENCE_LENGTH))
print('Shape of data tensor: ',data.shape)

#prepare embedding matrix

print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx.items():
	if i<MAX_VOCAB_SIZE:
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector


# load pre-trained word embedding into an Embedding layer
# trainable = False so the embeddings are fixed