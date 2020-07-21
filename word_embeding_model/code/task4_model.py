# from keras import initializers, regularizers, constraints
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical, plot_model
# from keras.models import Sequential, Model
# from keras.layers import Dense,Embedding,Activation,merge,Input,Lambda,Reshape,BatchNormalization
# from keras.layers import Convolution1D,Flatten,Dropout,MaxPool1D,GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
# from keras.layers import Conv1D,Bidirectional,LSTM,GRU, CuDNNLSTM, CuDNNGRU, concatenate
# from keras.engine.topology import Layer




import gensim
import pandas as pd
import numpy as np
import os
import sys
# sys.path.append("../targeted")
# from targeted_dropout import *

sys.path.append("../moel")
import f1
from LSTM import LSTM_Net
from LSTM_GRU_HANATTENTION import HAN_AttLayer, model_lstm_HAN, model_HAN
from LSTM_GRU_ATTENTION import model_lstm_atten
from CapsuleNet import lstm_conv_model




# import tensorflow as tf
# import keras.backend as K


# train_data = pd.read_csv("/home/baiyang/baiyang/semeval/task5/data/train.csv", engine='python')
# clean_questions = train_data.sample(frac=0.7, random_state=0, axis=0)
# test_data = train_data[~train_data.index.isin(clean_questions.index)]
clean_questions = pd.read_csv('/home/baiyang/baiyang/semeval/task4/data/train.csv', engine='python')
test_data = pd.read_csv('/home/baiyang/baiyang/semeval/task4/data/test.csv', engine='python')


# clean_questions.to_csv("train.csv", index=False)
# test_data.to_csv("test.csv", index=False)

# clean_questions=pd.read_table("data/train.txt")
# test_data=pd.read_table("data/testwithoutlabels.txt")

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = (clean_questions["sent0"]+'<eos>'+clean_questions["sent0"]).apply(tokenizer.tokenize)

print(clean_questions.head(5))

# test_data["tokens"] = clean_questions["sentence"].apply(tokenizer.tokenize)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
VOCAB_A = set(all_words)
VOCAB_B = list(VOCAB_A)
VOCAB = sorted  (VOCAB_B)
print("%s words total,with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

word2vec_path = "/home/baiyang/baiyang/code/crawl-300d-2M.vec"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = max(sentence_lengths) + 1
VOCAB_SIZE = len(VOCAB)

VALIDATION_SPLIT = 0.2
##############分词开始
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts((clean_questions["sent0"]+'<eos>'+clean_questions["sent1"]).tolist())
tokenizer.fit_on_texts((test_data["sent0"]+'<eos>'+test_data["sent1"]).tolist())
sequences_train = tokenizer.texts_to_sequences((clean_questions["sent0"]+'<eos>'+clean_questions["sent1"]).tolist())
sequences_test = tokenizer.texts_to_sequences((test_data["sent0"]+'<eos>'+test_data["sent1"]).tolist())

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

cnn_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
cnn_data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(clean_questions["label"]))
##
test_labels = test_data["label"]
y_true = np.array(test_labels)


indices = np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
cnn_data = cnn_data[indices]
labels = labels[indices]



num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

embedding_weights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, index in word_index.items():
    embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(embedding_weights.shape)



x_train = cnn_data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = cnn_data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# #lstm
model = lstm_conv_model(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index) + 1, EMBEDDING_DIM,
                len(list(clean_questions["label"].unique())))


from keras.callbacks import ModelCheckpoint

filepath = "/home/baiyang/baiyang/semeval/task4/mymodel/lstm_conv_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val), callbacks=callbacks_list)

scores = model.evaluate(x_val, y_val, verbose=0)
print("Acuracy: %.2f%%" % (scores[1] * 100))



def save_result(y_pred, file_name):
    result_df = pd.DataFrame({'Label': y_pred})
    result_df.to_csv(file_name, index=False)


model.load_weights(filepath)
#
y_pred = np.argmax(model.predict(cnn_data_test), axis=-1)
print(y_pred)
save_file = os.path.join('/home/baiyang/baiyang/semeval/task4/result', 'lstm_conv_model.csv')
save_result(y_pred, file_name=save_file)


# from sklearn.metrics import f1_score
# f1_macro = f1_score(y_true, y_pred,average='macro')
# print('f1_macro: {0}'.format(f1_macro))
