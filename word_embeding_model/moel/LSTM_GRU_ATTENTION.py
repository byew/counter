
import keras
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense,Embedding,Input
from keras.layers import Dropout,GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Bidirectional,LSTM,GRU, concatenate

import sys

from keras_targeted_dropout import TargetedDropout

# sys.path.append("../loss_funtion")
# from focal_loss import binary_focal_loss

from keras.layers import *
from keras.models import *



def f1(y_true, y_pred):
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



class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim



def model_lstm_atten(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    inp = Input(shape=(max_sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embeddings], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(LSTM(150, return_sequences=True))(x)
    y = Bidirectional(GRU(150, return_sequences=True))(x)

    atten_1 = Attention(max_sequence_length)(x)  # skip connect
    atten_2 = Attention(max_sequence_length)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)

    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(256, activation="relu")(conc)
    conc = Dropout(0.5)(conc)
    #conc = TargetedDropout(drop_rate=0.5, target_rate=0.5)(conc)
    #conc = TargetedDropout(drop_rate=0.5, target_rate=0.5)(conc)

    # conc = TargetedDropout(layer=keras.layers.Dense(units=2, activation='softmax'), drop_rate=0.5, target_rate=0.5, drop_patterns=['kernel'])(conc)
    outp = Dense(labels_index, activation="softmax")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss = binary_focal_loss(gamma=2, alpha=.25), optimizer='adam', metrics=['accuracy'])

    # model.compile(loss = binary_focal_loss(gamma=2, alpha=.25), optimizer='adam', metrics=[f1])

    plot_model(model, to_file="/home/baiyang/baiyang/semeval/task4/image/LSTM_GRU_ATTENTION.png")





    model.summary()

    return model