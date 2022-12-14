import numpy as np

from utils.LoadData import load
from utils.eval import eval
from utils.word2index import word2index

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


if __name__ == '__main__':
    data_train, data_test = load('./Tsinghua')
    BOW, pai = init(data_train)
    w2i = word2index(BOW)
    theta = get(BOW, w2i, data_train)
    acc = eval(data_test, pai, BOW, theta, t=1.5, mood=2)
    print("acc:{}".format(acc))
