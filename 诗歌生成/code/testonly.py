
# -*- coding:utf-8 -*-
import collections
import numpy as np
import pandas as pd
import time
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, RNN
from tensorflow.python.keras.utils import np_utils
import random
#from google.colab import drive
# drive.mount("/content/gdrive")
# 首先对诗词的文本格式进行处理

testonly = 1
epochs = 100
batch_size = 50
totallength = 32
less = 0
#filepath = "/content/gdrive/My Drive/yhtpoem/"
filepath = ""
lesser = ""
if less == 1:
    lesser = "less"


def text_process(filepath):
    seq_length = 8
    poems = []
    with open(filepath + "train.txt", "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            content = line.replace(' ', '')
            content = content.replace("\n", '')
            poems.append(content)
    all_word = []
    for poem in poems:
        for word in poem:
            all_word.append(word)
    # 对每个子进行计数
    counter = collections.Counter(all_word)
    # 对词进行排序，常用的词放在前面
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)

    words.append(' ')
    L = len(words)
    word_int_map = dict(zip(words, range(L)))  # 把词和词出现的次数存成字典

    X = []
    Y = []
    for poem in poems:
        for i in range(len(poem)-seq_length):
            sequence = poem[i:i + seq_length]
            label = poem[i + seq_length]
            X.append([word_int_map[char] for char in sequence])
            Y.append(word_int_map[label])

    X_modified = np.reshape(X, (len(X), seq_length, 1))
    X_modified = X_modified / float(len(words))
    numclass = len(words)
    Y_modified = np_utils.to_categorical(Y, numclass)

    X_valraw, Y_valraw = [], []
    valpoems = []
    with open(filepath + "val.txt", "r", encoding='utf-8', ) as valf:
        for valline in valf.readlines():
            valcontent = valline.replace(' ', '')
            valcontent = valcontent.replace("\n", '')
            valpoems.append(valcontent)
    for poem in valpoems:
        for i in range(len(poem)-seq_length):
            sequence = poem[i:i + seq_length]
            label = poem[i + seq_length]
            string_map = []
            for value in sequence:
                if value in word_int_map:
                    string_map.append(word_int_map[value])
                else:
                    string_map.append(random.randint(10, 100))
            X_valraw.append(string_map)
            if label in word_int_map:
                Y_valraw.append(word_int_map[label])
            else:
                Y_valraw.append(random.randint(10, 100))

    X_val = np.reshape(X_valraw, (len(X_valraw), seq_length, 1))
    X_val = X_val / float(len(words))
    Y_val = np_utils.to_categorical(Y_valraw, numclass)

    return X_modified, Y_modified, X_val, Y_val, words, word_int_map, seq_length


def output(word_int_map, words, place, modelname, writename):
    model = load_model(place + "result/" + modelname + ".h5")
    file = open(place + "data/" + lesser + "test.txt",
                "rb").read().decode('utf-8')
    X = file.replace(' ', '')
    X = X.split('\n')

    # 输出结果
    writer = open(place + "data/" + writename, "w")
    upwriter = open(place + "data/" + "generated_poem.txt", "w")
    for poem in X:
        if not poem:
            break
        # print(poem)
        string_mapped = []
        for value in poem:
            if value in word_int_map:
                string_mapped.append(word_int_map[value])
            else:
                string_mapped.append(random.randint(10, 100))
        full_string = list(poem)

    # generating characters(words)
        pre_length = totallength - seq_length
        for i in range(pre_length):
            x = np.reshape(string_mapped, (1, len(string_mapped), 1))
            x = x / float(len(words))

            pred_index = np.argmax(model.predict(x, verbose=0))
            #seq = [words[value] for value in string_mapped]
            full_string.append(words[pred_index])

            string_mapped.append(pred_index)
            string_mapped = string_mapped[1:len(string_mapped)]
        txt = u"".join(full_string)
        print(txt)
        writer.write(txt + "\n")
        upwriter.write(txt[8:32]+"\n")
    writer.close()
    upwriter.close()


if __name__ == "__main__":
    timeseg = []
    timeseg.append(time.time())
    X_modified, Y_modified, X_val, Y_val, words, word_int_map, seq_length = text_process(
        filepath + "data/")
    if testonly == 0:
        model = Sequential()
        model.add(LSTM(700, input_shape=(
            X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(700, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(700))
        model.add(Dropout(0.2))
        model.add(Dense(Y_modified.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        savepath = filepath + 'result/model-ep{epoch:03d}.h5'
        checkpoint = callbacks.ModelCheckpoint(
            savepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=10)
    timeseg.append(time.time())
    print("prepare time", timeseg[1] - timeseg[0])
    if testonly == 0:
        model.fit(X_modified, Y_modified, epochs=epochs,
                  batch_size=batch_size, callbacks=[checkpoint], validation_data=(X_val, Y_val), verbose=2)
        model.save(filepath + 'result/trymodel.h5')
    timeseg.append(time.time())
    print("train time", timeseg[2] - timeseg[1])
    epstr = "1010"
    '''
    for testepoch in range(7):
        if testepoch == 1:
            epstr = "320"
        elif testepoch == 2:
            epstr = "350"
        elif testepoch == 3:
            epstr = "380"
        elif testepoch == 4:
            epstr = "410"
        elif testepoch == 5:
            epstr = "440" 
        elif testepoch == 6:
            epstr = "470" 
    '''
    print("epoch:", epstr)
    output(word_int_map, words, filepath, "model-ep" + epstr, "output.txt")
    timeseg.append(time.time())
    print("test time:", timeseg[3] - timeseg[2])
    print("finish")
