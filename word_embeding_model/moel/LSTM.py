##########方法1 ——lstm##################
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import sys
sys.path.append('/home/lab1510/baiyang/code/task6/moel')
from f1 import precision, recall, fmeasure


def LSTM_Net(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = LSTM(256, dropout=0.2, recurrent_dropout=0.1)(embedded_sequences)

    preds = Dense(labels_index, activation='softmax')(x)
    model = Model(sequence_input, preds)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', precision, recall, fmeasure])



    return model