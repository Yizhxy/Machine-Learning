import argparse

import numpy as np
from libsvm.svm import *
from libsvm.svmutil import *

from FW import feature_select
from Bernoulli import word2index

Classes = ['电脑', '法律', '教育', '经济', '体育', '政治']
N = len(Classes)


def init(data):
    BOW = []
    Mu = np.zeros((N, 1))
    for i in range(N):
        Mu[i][0] = len(data[i])
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                word = data[i][j][k]
                if word not in BOW:
                    BOW.append(word)
    _ = np.sum(Mu)
    pai = Mu / _
    return BOW, pai


def load(path):
    data_train = []
    data_test = []
    for i in range(N):
        _data = []
        _path_train = path + './train./' + Classes[i] + '.txt'
        _path_test = path + './train./' + Classes[i] + '.txt'
        with open(_path_train, 'r', encoding="utf-8") as f:
            txt = f.read()
            txt = txt.split('<text>')
        for i in range(1, len(txt)):
            a = txt[i].split('</text>')[0].split('\n')[1]
            _data.append(a)
        data_train.append(_data)
        _data = []
        with open(_path_test, 'r', encoding="utf-8") as f:
            txt = f.read()
            txt = txt.split('<text>')
        for i in range(1, len(txt)):
            a = txt[i].split('</text>')[0].split('\n')[1]
            _data.append(a)
        data_test.append(_data)
    return data_train, data_test


def mySVM(x, y):
    problem = svm_problem(y, x, isKernel=True)
    # 采用核函数-t 2,5倍交叉验证模式-v 5
    options = svm_parameter('-s 0 -t 2 -v 5 -c 0.1')
    # 训练
    model = svm_train(problem, options)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--mood', default=2, type=int, help='data path')
    args = parser.parse_args()
    data_train, data_test = load('./Tsinghua')
    BOW, pai = init(data_train)
    w2i = word2index(BOW)
    data_num = 0
    for i in range(6):
        data_num += len(data_train[i])
    x = np.zeros((data_num, len(BOW)))
    y = np.zeros(data_num)
    ter = 0
    for i in range(6):
        features = feature_select(BOW, data_train[i], w2i, mood=args.mood)  # 所有词的TF值
        x[i] = features
        for j in range(len(data_train[i])):
            y[ter] = i
            ter += 1
    mySVM(x, y)
