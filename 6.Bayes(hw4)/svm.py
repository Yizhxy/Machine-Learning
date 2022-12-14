import argparse

import numpy as np
from libsvm.svm import *
from libsvm.svmutil import *

from Multinomial import load
from FW import feature_select
from Bernoulli import word2index
from Bernoulli import init


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
