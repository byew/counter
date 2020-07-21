
#######第三种模型#######


from keras import initializers, regularizers, constraints

from keras.models import  Model
from keras.layers import Dense, Embedding, Input, BatchNormalization
from keras.layers import Dropout,GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Bidirectional, LSTM, GRU, concatenate, CuDNNGRU, CuDNNLSTM, Conv1D
from keras.engine.topology import Layer

import keras.backend as K

import sys


import keras.backend as K

def lstm_conv_model(embedding_matrix, max_sequence_length, num_words,embedding_dim, labels_index):
    inp = Input(shape=(max_sequence_length, ))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.35)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(labels_index, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    return model