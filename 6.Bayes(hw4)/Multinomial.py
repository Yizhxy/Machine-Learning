import numpy as np

from utils.LoadData import load
from utils.eval import eval
from utils.word2index import word2index
from utils.init import init

Classes = ['电脑', '法律', '教育', '经济', '体育', '政治']
N = len(Classes)


def _a(data):
    BOW = []
    Mu = np.zeros((N, 1))
    for i in range(N):
        Mu[i][0] = len(data[i])
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                word = data[i][j][k]
                if word not in BOW:
                    BOW.append(word)
    theta = np.zeros((N, len(BOW)))
    for i in range(N):
        n1 = np.zeros(len(BOW))
        for j in range(len(BOW)):
            for k in range(len(data[i])):
                # n2 += len(data[i][k])
                str1 = data[i][k]
                n1[j] += str1.count(BOW[j])
        n2 = int(np.sum(n1))
        theta[i] = (n1 + 1) / (n2 + len(BOW))
    _ = np.sum(Mu)
    pai = Mu / _
    return theta, BOW, pai


def get(BOW, w2i, data, label, stop_words):
    theta = np.zeros((N, len(BOW)))
    for i in range(len(data)):
        _count = np.zeros((N, len(BOW)))
        for j in range(len(data[i])):
            word = data[i][j]
            if word in stop_words:
                continue
            index = w2i[word]
            _count[label[i], index] += 1
        theta += _count
    for i in range(N):
        theta[i] = (theta[i] + 1) / (sum(theta[i]) + len(BOW))
    return theta


if __name__ == '__main__':
    BOW, pai, data, label, num, stop_words = init()
    w2i = word2index(BOW)
    '''
    [:num]:train [num:]:test
    '''
    theta = get(BOW, w2i, data[:num], label, stop_words)
    acc = eval(data[num:], pai, BOW, theta, label[num:], stop_words, w2i, t=1000, mood=1, )
    print("acc:{}".format(acc))
