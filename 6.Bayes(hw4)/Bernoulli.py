import numpy as np


from utils.eval import eval
from utils.word2index import word2index
from utils.init import init

Classes = ['电脑', '法律', '教育', '经济', '体育', '政治']
N = len(Classes)


def get(BOW, w2i, data):
    theta = np.zeros((N, len(BOW)))
    for i in range(N):
        count = np.ones(len(BOW))
        for k in range(len(data[i])):
            str = data[i][k]
            used = np.zeros(len(BOW))
            for _ in range(len(str)):
                word = str[_]
                index = w2i[word]
                if used[index] == 0:
                    used[index] = 1
                    count[index] += 1.
        theta[i] = count / (len(data[i]) + 2.)
    return theta


def _a(BOW, w2i, data, label, stop_words):
    theta = np.zeros((N, len(BOW)))
    for i in range(len(data)):
        _count = np.zeros((N, len(BOW)))
        _used = np.zeros(len(BOW))
        for j in range(len(data[i])):
            word = data[i][j]
            if word in stop_words:
                continue
            index = w2i[word]
            if _used[index] == 0:
                _used[index] = 1
                _count[label[i], index] += 1
        theta += _count
    for i in range(N):
        theta[i] = (theta[i] + 1) / (label.count(i) + 2)
    return theta


if __name__ == '__main__':
    BOW, pai, data, label, num, stop_words = init()
    w2i = word2index(BOW)
    theta = _a(BOW, w2i, data[:num], label, stop_words)
    acc = eval(data[num:], pai, BOW, theta, label[num:], stop_words, w2i, t=2, mood=2, )
    print("acc:{}".format(acc))
