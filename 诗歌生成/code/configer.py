import collections
import numpy as np
import pandas as pd
from keras.utils import np_utils


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
            X_valraw.append([word_int_map[char] for char in sequence])
            Y_valraw.append(word_int_map[label])

    X_val = np.reshape(X_valraw, (len(X_valraw), seq_length, 1))
    X_val = X_val / float(len(words))
    Y_val = np_utils.to_categorical(Y_valraw, numclass)

    return X_modified, Y_modified, X_val, Y_val, words, word_int_map, seq_length
