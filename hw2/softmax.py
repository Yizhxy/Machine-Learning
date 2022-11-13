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


def load_data():
    X_train = np.loadtxt("./data/Exam/train/x.txt")
    Y_train = np.loadtxt("./data/Exam/train/y.txt", dtype=int)
    X_test = np.loadtxt("./data/Exam/test/x.txt")
    Y_test = np.loadtxt("./data/Exam/test/y.txt", dtype=int)
    return X_train, Y_train, X_test, Y_test


class Softmax(object):
    def __init__(self, X_train, Y_train):
        self.loss = None
        self.X_train = X_train
        self.Y_train = Y_train
        # M:特征数，N：样本数
        self.M = X_train.shape[1]
        self.N = X_train.shape[0]
        self.L = 2
        self.theta = -np.ones((self.L, self.M + 1))
        self.epoch = 500
        self.lr = 0.01

    def normalization(self):
        # 均值方差归一化
        mean = np.mean(self.X_train)
        variance = np.std(self.X_train)
        self.X_train = (self.X_train - mean) / variance
        self.X_train = np.insert(self.X_train, 0, values=1.0, axis=1)
        self.Y_train = self.Y_train.reshape(self.N, 1)
        self.M += 1

    def train(self):
        self.normalization()
        for i in range(self.epoch):
            g = self.gradient()
            self.theta += self.lr * g

    def shuffle(self):
        pass

    def gradient(self):
        g = np.zeros((self.L, self.M))
        y = self.Y_train
        for j in range(self.Y_train.shape[0]):
            _ = 0
            for l in range(self.L):
                _ += math.exp(np.dot(self.X_train[j], self.theta[l]))
            for l in range(self.L):
                if y[j] == l:
                    g[l] += (1 - np.dot(self.X_train[j], self.theta[l])/_) * self.X_train[j]
                else:
                    g[l] += (0 - np.dot(self.X_train[j], self.theta[l])/_) * self.X_train[j]
        return g


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    model = Softmax(X_train, Y_train)
    model.train()
