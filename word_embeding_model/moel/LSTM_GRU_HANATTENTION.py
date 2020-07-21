#######第三种模型#######

from keras import initializers, regularizers, constraints

from keras.models import  Model
from keras.layers import Dense, Embedding, Input, BatchNormalization
from keras.layers import Dropout,GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Bidirectional,LSTM,GRU,concatenate, CuDNNGRU, CuDNNLSTM
from keras.engine.topology import Layer

import keras.backend as K

import sys

from keras_targeted_dropout import TargetedDropout
from tensorflow.python.keras.utils import plot_model



sys.path.append("../loss_funtion")
from focal_loss import binary_focal_loss

from on_lstm_keras import ONLSTM



class HAN_AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(HAN_AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)  # (x, 40, 1)
        uit = K.squeeze(uit, -1)  # (x, 40)
        uit = uit + self.b  # (x, 40) + (40,)
        uit = K.tanh(uit)  # (x, 40)

        ait = uit * self.u  # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait)  # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())  # (x, 40)
            ait = mask * ait  # (x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def f1(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model_HAN(embedding_matrix, max_sequence_length, num_words, embedding_dim, labels_index):
    inp = Input(shape=(max_sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    y = Bidirectional(GRU(40, return_sequences=True))(x)

    atten_1 = HAN_AttLayer()(x)  # skip connect
    atten_2 = HAN_AttLayer()(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)

    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    # conc = TargetedDropout(drop_rate=0.5, target_rate=0.5)(conc)
    outp = Dense(labels_index, activation="softmax")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.compile(loss=binary_focal_loss(gamma=2, alpha=.25), optimizer='adam', metrics=['accuracy'])
    # plot_model(model, to_file="/home/baiyang/baiyang/semeval/task4/image/LSTM_GRU_han.png")


    return model


# def model_lstm_HAN(embedding_matrix, max_sequence_length, num_words,embedding_dim, labels_index):
#     inp = Input(shape=(max_sequence_length,))
#     x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(inp)
#     x = SpatialDropout1D(0.1)(x)
#     # x = LSTM(40, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(x)
#     x = ONLSTM(128, 4, dropconnect=0.25, return_sequences=True)(x)
#
#     x = Dropout(0.25)(x)
#     attention = HAN_AttLayer()(x)
#     fc = Dense(256, activation='relu')(attention)
#     fc = Dropout(0.25)(fc)
#     fc = BatchNormalization()(fc)
#
#     outp = Dense(labels_index, activation="sofmax")(fc)
#
#     model = Model(inputs=inp, outputs=outp)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
#     plot_model(model, to_file="/home/lab1510/baiyang/semeval/task5/image/LSTM_GRU_ATTENTION.png")
#
#
#     return model



def model_lstm_HAN(embedding_matrix, max_sequence_length, num_words,embedding_dim, labels_index):
    inp = Input(shape=(max_sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = LSTM(40, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(x)
    # x = ONLSTM(128, 4, dropconnect=0.25, return_sequences=True)(x)

    x = Dropout(0.25)(x)
    attention = HAN_AttLayer()(x)
    fc = Dense(256, activation='relu')(attention)
    fc = Dropout(0.25)(fc)
    fc = BatchNormalization()(fc)

    outp = Dense(labels_index, activation="sigmoid")(fc)
    # outp = Dense(1, activation="sigmoid")(fc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    plot_model(model, to_file="/home/baiyang/baiyang/semeval/task4/image/LSTM_GRU_HANATTENTION.png")


    return model


def model_lstm_HN(embedding_matrix, max_sequence_length, num_words,embedding_dim, labels_index):
    inp = Input(shape=(max_sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = LSTM(40, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(x)
    x = Dropout(0.25)(x)
    attention = HAN_AttLayer()(x)
    fc = Dense(256, activation='relu')(attention)
    fc = Dropout(0.25)(fc)
    fc = BatchNormalization()(fc)

    outp = Dense(labels_index, activation="sigmoid")(fc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model

