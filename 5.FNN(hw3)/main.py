import numpy as np
import matplotlib.pyplot as plt

import argparse
import math
import random


class Linear:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.x = None
        self.theta = np.zeros((input_dim, output_dim))
        self.bias = np.ones((1, output_dim))
        self.lr = lr

    def forward(self, x):
        # param x: [B,input_dim]
        self.x = x
        return x @ self.theta + self.bias

    def bp(self, gradient):
        # param loss:[B,output_dim]
        self.theta -= self.lr * (self.x.T @ gradient)
        self.bias -= self.lr * np.ones((1, gradient.shape[0])) @ gradient
        return gradient @ self.theta.T


class Softmax:
    def __init__(self, input_dim, output_dim):
        self.linear_layer = Linear(input_dim, output_dim)
        self.x = None
        self.s_matrix = np.zeros((output_dim, output_dim))

    def forward(self, x):
        self.x = x
        x = self.linear_layer.forward(x)
        _ = np.zeros(x.shape)
        for i in range(_.shape[0]):
            _[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
        return _

    def bp(self, gradient):
        _ = self.linear_layer.bp(gradient)
        return _


class Sigmoid:
    def __init__(self, input_dim, output_dim=1):
        self.theta = np.zeros((input_dim, output_dim))

    def forward(self, x):
        return 1 / (1 + np.exp(-x @ self.theta))


def load_data(path):
    X_train = np.loadtxt(path + "/train/x.txt")
    Y_train = np.loadtxt(path + "/train/y.txt", dtype=int)
    X_test = np.loadtxt(path + "/test/x.txt")
    Y_test = np.loadtxt(path + "/test/y.txt", dtype=int)
    return X_train, Y_train, X_test, Y_test


class FNN:
    def __init__(self, opt, X_train, Y_train, X_test, Y_test, epoch, lr, is_decay, lr_decay, batch_size):
        self.loss = None
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        # M:特征数，N:样本数 L:类别数
        self.M = X_train.shape[1]
        self.N = X_train.shape[0]
        self.L = np.max(self.Y_train) + 1
        self.epoch = epoch
        self.lr = lr
        self.is_decay = is_decay
        self.lr_decay = lr_decay
        self.batch_size = batch_size if batch_size > 0 else -1
        self.normalization()
        self.model = self.init_model()

    def init_model(self):
        creat_model = []
        creat_model.append(Linear(self.M, 2 * self.M - 1))
        creat_model.append(Softmax(2 * self.M - 1, self.L))
        return creat_model

    def normalization(self):
        mean = np.mean(self.X_train)
        variance = np.std(self.X_train)
        self.X_train = (self.X_train - mean) / variance
        # self.X_train = np.insert(self.X_train, 0, values=1.0, axis=1)
        self.X_test = (self.X_test - mean) / variance
        # self.X_test = np.insert(self.X_test, 0, values=1.0, axis=1)
        self.Y_train = self.Y_train.reshape(self.N, 1)
        self.Y_test = self.Y_test.reshape(self.X_test.shape[0], 1)

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

    def forward(self, x):
        _ = x
        for i in range(len(self.model)):
            _ = self.model[i].forward(_)
        return _

    def bp(self, x, y):
        _y = np.zeros((self.batch_size, self.L))
        for i in range(self.batch_size):
            _ = int(y[i][0])
            _y[i][_] = 1
        _ = x - _y
        for i in range(len(self.model) - 1, -1, -1):
            _ = self.model[i].bp(_)

    def eval(self):
        x = self.X_test
        y = self.Y_test
        T = 0
        y_hat = self.forward(x)
        for i in range(x.shape[0]):
            predict = np.argmax(y_hat[i])
            if predict == y[i]:
                T += 1
        print(T / x.shape[0])

    def train(self):
        for _ in range(self.epoch):
            x, y = self.shuffle()
            _ = self.forward(x)
            self.bp(_, y)
            self.eval()
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data_path', default='./data/Exam', help='data path')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=800, type=int, help='epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='learning rate decay')
    parser.add_argument('--is_decay', default=True, type=bool, help='choose to use decay')
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--fun', default=['linear'], type=list)
    parser.add_argument('--input_features', default=['2'], type=list)
    parser.add_argument('--output_features', default=['1'], type=list)
    args = parser.parse_args()
    X_train, Y_train, X_test, Y_test = load_data(args.data_path)
    model = FNN(args, X_train, Y_train, X_test, Y_test, epoch=args.epoch, lr=args.lr, is_decay=args.is_decay,
                lr_decay=args.lr_decay,
                batch_size=args.batch_size)
    model.train()
