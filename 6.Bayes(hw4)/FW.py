# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import numpy as np
from utils.LoadData import load
from utils.word2index import word2index
from Bernoulli import init


def feature_select(BOW, list_words, w2i, mood):
    # 总词频统计
    frequency = np.zeros(len(BOW))
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            index = w2i[i]
            if mood == 1:
                frequency[index] = 1
                doc_frequency[i] += 1
            elif mood == 2:
                frequency[index] += 1
    if mood == 1:
        return frequency
    else:
        _ = np.sum(frequency)
        frequency = frequency / _
        return frequency


if __name__ == '__main__':
    data_train, data_test = load('./Tsinghua')
    BOW, pai = init(data_train)
    w2i = word2index(BOW)
    for i in range(6):
        features = feature_select(BOW, data_train[i], w2i, mood=2)  # 所有词的TF值
        print(features)
