# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import numpy as np
from utils.eval import eval
from utils.word2index import word2index
from utils.init import init


def feature_select(BOW, list_words, w2i, mood):
    # 总词频统计
    frequency = np.zeros(len(BOW))
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            if i in BOW:
                index = w2i[i]
            else:
                continue
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
    BOW, pai, data, label, num, stop_words = init()
    w2i = word2index(BOW)
    feature = feature_select(BOW, data, w2i, mood=1)
