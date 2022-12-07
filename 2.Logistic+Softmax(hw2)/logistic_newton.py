import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np


def load_data():
    X_train = np.loadtxt("./data/Exam/train/x.txt")
    Y_train = np.loadtxt("./data/Exam/train/y.txt", dtype=int)
    X_test = np.loadtxt("./data/Exam/test/x.txt")
    Y_test = np.loadtxt("./data/Exam/test/y.txt", dtype=int)
    return X_train, Y_train, X_test, Y_test


class Logistic(object):
    def __init__(self, X_train, Y_train):
        self.loss = None
        self.X_train = X_train
        self.Y_train = Y_train
        # M:特征数，N：样本数
        self.M = X_train.shape[1]
        self.N = X_train.shape[0]
        self.normalization()
        self.theta = -np.ones((self.M, 1))

    def normalization(self):
        # 均值方差归一化
        mean = np.mean(self.X_train)
        variance = np.std(self.X_train)
        self.X_train = (self.X_train - mean) / variance
        self.X_train = np.insert(self.X_train, 0, values=1.0, axis=1)
        self.Y_train = self.Y_train.reshape(self.N, 1)
        self.M += 1

    def sigmoid(self, X):
        eta = -np.dot(X, self.theta)  # N*1
        H = np.exp(eta)
        H = 1.0 / (1.0 + H)
        return H

    def Newton_method(self):
        plt.ion()
        for i in range(100):
            H = self.sigmoid(self.X_train)
            J = np.dot(self.X_train.T, (H - self.Y_train))  # M*1
            Hession = np.dot(H.T, self.X_train).dot(self.X_train.T).dot((1.0 - H)) / self.N
            self.theta -= np.dot(J, np.linalg.inv(Hession))
            loss = -np.sum(self.Y_train * np.log(H) + (1.0 - self.Y_train) * np.log(1 - H)) / self.N
            plt.xlim((-2, 2))
            plt.ylim((-2, 2))
            for j in range(self.X_train.shape[0]):
                if self.Y_train[j] == 0:
                    plt.plot(self.X_train[j][1], self.X_train[j][2], 'or')
                else:
                    plt.plot(self.X_train[j][1], self.X_train[j][2], 'ob')
            p1 = -(self.theta[0][0] + self.theta[1][0] * 2) / self.theta[2][0]
            p2 = -(self.theta[0][0] + self.theta[1][0] * -2) / self.theta[2][0]
            plt.plot([2, -2], [p1, p2])
            plt.pause(0.001)
            plt.clf()
            print("iter: %d, loss: %f" % (i, loss))
        plt.ioff()
        plt.show()

    def train(self):
        self.Newton_method()


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    model=Logistic(X_train, Y_train)
    model.train()