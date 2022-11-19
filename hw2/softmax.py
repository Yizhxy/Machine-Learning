import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def load_data(path):
    X_train = np.loadtxt(path + "/train/x.txt")
    Y_train = np.loadtxt(path + "/train/y.txt", dtype=int)
    X_test = np.loadtxt(path + "/test/x.txt")
    Y_test = np.loadtxt(path + "/test/y.txt", dtype=int)
    return X_train, Y_train, X_test, Y_test


class Softmax(object):
    def __init__(self, X_train, Y_train, X_test, Y_test, epoch, lr, is_decay, lr_decay, batch_size):
        self.loss = None
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        # M:特征数，N:样本数 L:类别数
        self.M = X_train.shape[1]
        self.N = X_train.shape[0]
        self.L = np.max(self.Y_train) + 1
        self.theta = -np.ones((self.L, self.M + 1))
        self.epoch = epoch
        self.lr = lr
        self.is_decay = is_decay
        self.lr_decay = lr_decay
        self.batch_size = batch_size if batch_size > 0 else -1
        self.normalization()

    def normalization(self):
        # 均值方差归一化
        mean = np.mean(self.X_train)
        variance = np.std(self.X_train)
        self.X_train = (self.X_train - mean) / variance
        self.X_train = np.insert(self.X_train, 0, values=1.0, axis=1)
        self.X_test = (self.X_test - mean) / variance
        self.X_test = np.insert(self.X_test, 0, values=1.0, axis=1)
        self.Y_train = self.Y_train.reshape(self.N, 1)
        self.Y_test = self.Y_test.reshape(self.X_test.shape[0], 1)
        self.M += 1

    def train(self):
        plt.ion()
        plt.figure(figsize=(10, 3))
        _acc_train, _acc_test, _i = [], [], []
        _loss_train, _loss_test = [], []
        for i in range(self.epoch):
            if i % 100 == 0 and self.is_decay:
                self.lr = self.lr * self.lr_decay
            acc_on_train = self.validate(self.X_train, self.Y_train)
            acc_on_test = self.validate(self.X_test, self.Y_test)
            _i.append(i)
            _acc_train.append(acc_on_train)
            _acc_test.append(acc_on_test)
            plt.cla()
            plt.subplot(1, 3, 2)
            plt.xlim((0, self.epoch))
            plt.ylim((0, 1))
            plt.xlabel("epoch")
            plt.ylabel("acc")
            line1, = plt.plot(_i, _acc_train, 'r')
            line2, = plt.plot(_i, _acc_test, 'b')
            plt.legend((line1, line2), ('train', 'test'), loc='lower right', fontsize='small')
            print("epoch:{} acc:{}".format(i, acc_on_train))

            g, loss1, loss2 = self.gradient()
            # print("train loss:{} test loss:{}".format(loss1,loss2))
            plt.subplot(1, 3, 1)
            plt.title("gd with lr decay" if self.is_decay else "gd without lr decay")
            plt.xlim((0, self.epoch))
            plt.ylim((-1, 0))
            plt.xlabel("epoch")
            plt.ylabel("avg logL")
            _loss_train.append(loss1)
            _loss_test.append(loss2)
            line1, = plt.plot(_i, _loss_train, 'r')
            line2, = plt.plot(_i, _loss_test, 'b')
            plt.legend((line1, line2), ('train', 'test'), loc='lower right', fontsize='small')
            self.theta += self.lr * g

            plt.subplot(1, 3, 3)
            plt.xlim((-3, 3))
            plt.ylim((-3, 3))
            l1, l2, l3, l4, l5, l6 = None, None, None, None, None, None
            for i in range(self.X_train.shape[0]):
                # if you want to train on your own dataset you should change this part
                if self.Y_train[i] == 0:
                    l1, = plt.plot(self.X_train[i][1], self.X_train[i][2], 'or')
                elif self.Y_train[i] == 1:
                    l2, = plt.plot(self.X_train[i][1], self.X_train[i][2], 'ob')
                elif self.Y_train[i] == 2:
                    l3, = plt.plot(self.X_train[i][1], self.X_train[i][2], 'oy')
            for i in range(self.X_test.shape[0]):
                # if you want to train on your own dataset you should change this part
                if self.Y_test[i] == 0:
                    l4, = plt.plot(self.X_test[i][1], self.X_test[i][2], 'or', alpha=0.2)
                elif self.Y_test[i] == 1:
                    l5, = plt.plot(self.X_test[i][1], self.X_test[i][2], 'ob', alpha=0.2)
                elif self.Y_test[i] == 2:
                    l6, = plt.plot(self.X_test[i][1], self.X_test[i][2], 'oy', alpha=0.2)
            if self.L == 2:
                plt.legend((l1, l2, l4, l5), ('negative', 'positive', 'negative_on_test', 'positive_on_test'),
                           loc='lower right', fontsize='small')
            if self.L == 3:
                plt.legend((l1, l2, l3, l4, l5, l6), ('class 1', 'class 2', 'class 3', 'class 1 on test',
                                                      'class 2 on test', 'class 3 on test'),
                           loc='upper left', fontsize='small')
            p1 = np.zeros((self.L, 2))
            p2 = np.zeros((self.L, 2))
            for j in range(self.L):
                if j == 0:
                    opts = 'r'
                elif j == 1:
                    opts = 'b'
                elif j == 2:
                    opts = 'y'
                p1[j][0] = 3
                p2[j][0] = -3
                p1[j][1] = (self.theta[j][0] + self.theta[j][1] * p1[j][0]) / self.theta[j][2]
                p2[j][1] = (self.theta[j][0] + self.theta[j][1] * p2[j][0]) / self.theta[j][2]
                plt.plot([p1[j][0], p2[j][0]], [p1[j][1], p2[j][1]], color=opts)
            plt.pause(0.001)
            plt.clf()
        plt.ioff()
        plt.show()
        plt.pause(5)
        plt.close('all')

    def shuffle(self):
        # 实现的功能并非shuffle，而是随机挑选数据
        if self.batch_size == -1:
            return self.X_train, self.Y_train
        _x = np.zeros((self.batch_size, self.M))
        _y = np.zeros((self.batch_size, 1))
        max_index = self.Y_train.shape[0] - 1
        for i in range(self.batch_size):
            _ = random.randint(0, max_index)
            _x[i] = self.X_train[_]
            _y[i] = self.Y_train[_]
        return _x, _y

    def gradient(self):
        g = np.zeros((self.L, self.M))
        loss1, loss2 = 0, 0
        x, y = self.shuffle()
        for j in range(y.shape[0]):
            _ = 0
            for l in range(self.L):
                _ += math.exp(np.dot(x[j], self.theta[l]))
            for l in range(self.L):
                if y[j] == l:
                    g[l] += (1 - math.exp(np.dot(x[j], self.theta[l])) / _) * x[j]
                else:
                    g[l] += (0 - math.exp(np.dot(x[j], self.theta[l])) / _) * x[j]
        for j in range(self.X_train.shape[0]):
            _ = 0
            for l in range(self.L):
                _ += math.exp(np.dot(self.X_train[j], self.theta[l]))
            for l in range(self.L):
                if self.Y_train[j] == l:
                    loss1 += math.log(math.exp(np.dot(self.X_train[j], self.theta[l])) / _)
        for j in range(self.X_test.shape[0]):
            _ = 0
            for l in range(self.L):
                _ += math.exp(np.dot(self.X_test[j], self.theta[l]))
            for l in range(self.L):
                if self.Y_test[j] == l:
                    loss2 += math.log(math.exp(np.dot(self.X_test[j], self.theta[l])) / _)
        loss1 /= self.X_train.shape[0]
        loss2 /= self.X_test.shape[0]
        return g, loss1, loss2

    def validate(self, X, Y):
        TP = 0
        for i in range(Y.shape[0]):
            index = -1
            maxp = 0
            _ = 0
            for j in range(self.L):
                _ += np.exp(np.dot(X[j], self.theta[j]))
            for j in range(self.L):
                p = np.exp(np.dot(X[i], self.theta[j])) / _
                if p > maxp:
                    maxp = p
                    index = j
            if index == Y[i]:
                TP += 1
        acc = TP / Y.shape[0]
        return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data_path', default='./data/Iris', help='data path')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=300, type=int, help='epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='learning rate decay')
    parser.add_argument('--is_decay', default=True, type=bool, help='choose to use decay')
    args = parser.parse_args()
    X_train, Y_train, X_test, Y_test = load_data(args.data_path)
    model = Softmax(X_train, Y_train, X_test, Y_test, epoch=args.epoch, lr=args.lr, is_decay=args.is_decay,
                    lr_decay=args.lr_decay,
                    batch_size=args.batch_size)
    model.train()
