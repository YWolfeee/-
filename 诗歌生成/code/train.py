# -*- coding:utf-8 -*-
import collections
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RNN
from keras.utils import np_utils
import random
import configer
#from google.colab import drive

#drive.mount("/content/gdrive")
# 首先对诗词的文本格式进行处理

epochs = 1
batch_size = 50
totallength = 32
less = 1
#filepath = "/content/gdrive/My Drive/yhtpoem/"
filepath = ""
lesser = ""
if less == 1:
    lesser = "less"

if __name__ == "__main__":

    X_modified, Y_modified, X_val, Y_val, words, word_int_map, seq_length = configer.text_process(
        filepath + "data/" + lesser)
    model = Sequential()
    model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(700, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(700))
    model.add(Dropout(0.2))
    model.add(Dense(Y_modified.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_modified, Y_modified, epochs=epochs,
              batch_size=batch_size, validation_data=(X_val, Y_val), verbose=2)
    model.save(filepath + 'result/trymodel.h5')

    print("finish")
