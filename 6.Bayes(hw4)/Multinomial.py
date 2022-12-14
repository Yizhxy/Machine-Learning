import numpy as np

from utils.LoadData import load
from utils.eval import eval

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


if __name__ == '__main__':
    data_train, data_test = load('./Tsinghua')
    theta, BOW, pai = init(data_train)
    acc = eval(data_train, pai, BOW, theta, t=100, mood=1)
    print("acc:{}".format(acc))
