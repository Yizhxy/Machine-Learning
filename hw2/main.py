import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    if x > 100:
        return 1
    elif x < -100:
        return 0
    else:
        return 1.0 / (1 + math.exp(-x))


def get_data(path):
    data = path
    with open(data + "/x.txt", "r") as f:
        _ = f.readlines()
        x = []
        for i in range(len(_)):
            j = _[i].split(' ')
            x.append(float(j[0]))
            x.append(float(j[1]))
            x.append(1.)
        x = np.array(x).reshape(-1, 3)
    with open(data + "/y.txt", "r") as f:
        _ = f.readlines()
        y = []
        for i in range(len(_)):
            y.append(int(_[i].split('\n')[0]))
        y = np.array(y)
        assert x.shape[0] == y.shape[0], "the shape of x and y are inconsistent"
        x = np.mat(x)
        x0_min = x[:, 0].min()
        x0_max = x[:, 0].max()
        x1_min = x[:, 1].min()
        x1_max = x[:, 1].max()
    return x, y, x0_min, x1_min, x0_max, x1_max


class Logistic:
    def __init__(self, args):
        self.x_train, self.y_train, self.x0_min, self.x1_min, self.x0_max, self.x1_max = get_data(args.data + '/train')
        self.x_train[:, 0] = (self.x_train[:, 0] - self.x0_min) / (self.x0_max - self.x0_min)
        self.x_train[:, 1] = (self.x_train[:, 1] - self.x1_min) / (self.x1_max - self.x1_min)
        self.x_test, self.y_test, _, __, _, _ = get_data(args.data + '/test')
        self.x_test[:, 0] = (self.x_test[:, 0] - self.x0_min) / (self.x0_max - self.x0_min)
        self.x_test[:, 1] = (self.x_test[:, 1] - self.x1_min) / (self.x1_max - self.x1_min)
        self.batch_size = args.batch_size
        self.theta = [-1.0, 1.0, 1.0]
        self.bs = 0
        self.lr = args.lr
        self.epoch = args.epoch
        self.lr_decay = args.lr_decay
        self.is_decay = args.is_decay

    def shuffle(self):
        # 实现的功能并非shuffle，而是随机挑选数据
        _x, _y = [], []
        if self.batch_size == -1:
            for i in range(self.y_train.shape[0]):
                _x.append(self.x_train[i])
                _y.append(self.y_train[i])
            return _x, _y
        batch_size = self.batch_size
        max_index = int(self.y_train.shape[0]) - 1
        _x, _y = [], []
        for i in range(batch_size):
            _ = random.randint(0, max_index)
            _x.append(self.x_train[_])
            _y.append(self.y_train[_])
        test = _x[0][0, 1]
        return _x, _y

    def forward(self):
        for i in range(self.y_train.shape[0]):
            plt.plot(self.x_train[i, 0], self.x_train[i, 1], 'ob' if self.y_train[i] == 1 else 'or')
        plt.show()

    def gradient(self):
        g0 = 0
        g1 = 0
        g2 = 0
        x, y = self.shuffle()
        for i in range(len(y)):
            _ = (y[i] - sigmoid(self.theta[2] +
                                self.theta[0] * x[i][0, 0] + self.theta[1] * x[i][0, 1]))
            g0 += _ * x[i][0, 0]
            g1 += _ * x[i][0, 1]
            g2 += _
        return g0, g1, g2

    def validate(self):
        theta = self.theta
        x_test = self.x_test
        y_test = self.y_test
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(y_test)):
            predict = 1 if theta[0] * x_test[i, 0] + theta[1] * x_test[i, 1] + theta[2] >= 0 else 0
            label = y_test[i]
            if predict == 0 and label == 0:
                TN += 1
            elif predict == 0 and label == 1:
                FN += 1
            elif predict == 1 and label == 0:
                FP += 1
            elif predict == 1 and label == 1:
                TP += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        return precision, recall, F, acc

    def plot(self):
        pass

    def train(self):
        # if self.batch_size != -1:
        # self.shuffle()
        plt.ion()
        plt.figure(figsize=(10, 5))
        precision, recall, F, acc = [], [], [], []
        _loss = []
        _i = []

        for i in range(self.epoch):
            _i.append(i)
            if i % 100 == 0 and self.is_decay:
                self.lr = self.lr * self.lr_decay
            g0, g1, g2 = self.gradient()
            _loss.append(g0)
            self.theta[0] += g0 * self.lr
            self.theta[1] += g1 * self.lr
            self.theta[2] += g2 * self.lr
            _precision, _recall, _F, _acc = self.validate()
            print("epoch:{} acc:{}".format(i, _acc))
            precision.append(_precision)
            recall.append(_recall)
            F.append(_F)
            acc.append(_acc)
            plt.cla()
            plt.subplot(1, 2, 1)
            plt.title("gd with lr decay" if self.is_decay else "gd without lr decay")
            plt.xlim((0, 1000))
            plt.ylim((0, 5))
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(_i, _loss)
            # plt.figure(2)
            plt.subplot(1, 2, 2)
            plt.title("metrics")
            plt.xlim((0, 1000))
            plt.ylim((0, 1))
            plt.xlabel("epoch")
            plt.ylabel("per")
            l1, = plt.plot(_i, precision, color='green')
            l2, = plt.plot(_i, recall, color='blueviolet')
            l3, = plt.plot(_i, F, color='orangered')
            l4, = plt.plot(_i, acc, color='red', label='acc')
            plt.legend((l1, l2, l3, l4), ('precision', 'recall', 'F', 'acc'), loc='lower right', shadow=True)
            plt.pause(0.001)
            plt.clf()
        plt.ioff()
        plt.show()
        plt.pause(5)
        plt.close('all')

    def predict(self):
        pass


if __name__ == '__main__':
    get_data('./data/Exam/train')
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data', default='./data/Exam', help='data path')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=1000, type=int, help='epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='learning rate decay')
    parser.add_argument('--is_decay', default=True, type=bool, help='choose to use decay')
    args = parser.parse_args()
    model = Logistic(args)
    model.train()
