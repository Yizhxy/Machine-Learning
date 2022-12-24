import numpy as np
import libsvm
from libsvm.svm import *
from libsvm.svmutil import *
import json

import re

Classes = ['电脑', '法律', '教育', '经济', '体育', '政治']
N = len(Classes)


def create(data, stop_words, w2i):
    Feature_bool = np.zeros(len(w2i))
    Feature_tf = np.zeros(len(w2i))
    xiangliang = np.zeros(len(w2i))
    for j in range(len(data)):
        if data[j] in stop_words:
            continue
        else:
            index = w2i[data[j]]
            xiangliang[index] += 1
            if Feature_bool[index] == 0:
                Feature_bool[index] = 1
            continue
    fenmu = sum(xiangliang)
    for j in range(len(data)):
        Feature_tf[j] = xiangliang[j] * 100 / fenmu
    return xiangliang, Feature_bool, Feature_tf


def word2index(BOW):
    word2index_map = {}
    for i in range(len(BOW)):
        word = BOW[i]
        word2index_map.setdefault(word, i)
    return word2index_map


def savejson(BOW, data, label, num, stop_words):
    filename = 'BOW.json'
    with open(filename, 'w') as file_obj:
        json.dump(BOW, file_obj)
    filename = 'data.json'
    with open(filename, 'w') as file_obj:
        json.dump(data, file_obj)
    filename = 'label.json'
    with open(filename, 'w') as file_obj:
        json.dump(label, file_obj)
    filename = 'num.json'
    with open(filename, 'w') as file_obj:
        json.dump(num, file_obj)
    filename = 'stop_words.json'
    with open(filename, 'w') as file_obj:
        json.dump(stop_words, file_obj)


def loadjson():
    filename = 'BOW.json'
    with open(filename, 'r', encoding='utf8') as fp:
        BOW = json.load(fp)

    filename = 'data.json'
    with open(filename, 'r', encoding='utf8') as fp:
        data = json.load(fp)

    filename = 'label.json'
    with open(filename, 'r', encoding='utf8') as fp:
        label = json.load(fp)

    filename = 'num.json'
    with open(filename, 'r', encoding='utf8') as fp:
        num = json.load(fp)

    filename = 'stop_words.json'
    with open(filename, 'r', encoding='utf8') as fp:
        stop_words = json.load(fp)
    return BOW, data, label, num, stop_words


if __name__ == '__main__':

    BOW, data, label, num, stop_words = loadjson()
    w2i = word2index(BOW)

    xiangliang = np.zeros((len(data), len(BOW)))

    bool = np.zeros((len(data), len(BOW)))
    tf = np.zeros((len(data), len(BOW)))
    labelbool = []
    for i in range(0, len(data)):
        print('load data' + str((i + 1) / len(data)))
        xiangliang[i], bool[i], tf[i] = create(data[i], stop_words, w2i)

    print("tf特征表示")
    options = svm_parameter('-s 0 -t 2 -v 5 -c 0.2')
    problemtf = svm_problem(label[num:], tf[num:], isKernel=True)
    modeltf = svm_train(problemtf, options)
    print("bool特征表示")
    options = svm_parameter('-s 0 -t 0 -v 5 -c 0.1')
    problembool = svm_problem(label[num:], bool[num:], isKernel=True)
    modelbool = svm_train(problembool, options)
