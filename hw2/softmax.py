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
    return X_test, Y_test, X_test, Y_test


class Softmax(object):
    def __init__(self, X_train, Y_train, epoch, lr, is_decay, lr_decay, batch_size):
        self.loss = None
        self.X_train = X_train
        self.Y_train = Y_train
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
        self.Y_train = self.Y_train.reshape(self.N, 1)
        self.M += 1

    def train(self):
        plt.ion()
        for i in range(self.epoch):
            if i % 100 == 0 and self.is_decay:
                self.lr = self.lr * self.lr_decay
            g = self.gradient()
            self.theta += self.lr * g
            plt.xlim((-3, 3))
            plt.ylim((-3, 3))
            for i in range(self.X_train.shape[0]):
                # if you want to train on your own dataset you should change this part
                if self.Y_train[i] == 0:
                    plt.plot(self.X_train[i][1], self.X_train[i][2], 'or')
                elif self.Y_train[i] == 1:
                    plt.plot(self.X_train[i][1], self.X_train[i][2], 'ob')
                elif self.Y_train[i] == 2:
                    plt.plot(self.X_train[i][1], self.X_train[i][2], 'oy')
            p1 = np.zeros((self.L, 2))
            p2 = np.zeros((self.L, 2))
            for j in range(self.L):
                if j == 0:
                    opts = 'y'
                elif j == 1:
                    opts = 'b'
                elif j == 2:
                    opts = 'r'
                p1[j][0] = 3
                p2[j][0] = -3
                p1[j][1] = (self.theta[j][0] + self.theta[j][1] * p1[j][0]) / self.theta[j][2]
                p2[j][1] = (self.theta[j][0] + self.theta[j][1] * p2[j][0]) / self.theta[j][2]
                plt.plot([p1[j][0], p2[j][0]], [p1[j][1], p2[j][1]], opts)
            plt.pause(0.001)
            plt.clf()
        plt.ioff()
        plt.show()

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
        x, y = self.shuffle()
        for j in range(y.shape[0]):
            _ = 0
            for l in range(self.L):
                _ += math.exp(np.dot(x[j], self.theta[l]))
            for l in range(self.L):
                if y[j] == l:
                    g[l] += (1 - np.dot(x[j], self.theta[l]) / _) * x[j]
                else:
                    g[l] += (0 - np.dot(x[j], self.theta[l]) / _) * x[j]
        return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data_path', default='./data/Iris', help='data path')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=1000, type=int, help='epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='learning rate decay')
    parser.add_argument('--is_decay', default=True, type=bool, help='choose to use decay')
    args = parser.parse_args()
    X_train, Y_train, X_test, Y_test = load_data(args.data_path)
    model = Softmax(X_train, Y_train, epoch=args.epoch, lr=args.lr, is_decay=args.is_decay, lr_decay=args.lr_decay,
                    batch_size=args.batch_size)
    model.train()
