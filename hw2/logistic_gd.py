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
    assert X_train.shape[0] == Y_train.shape[0] and X_test.shape[0] == Y_test.shape[0], "shape error"
    return X_train, Y_train, X_test, Y_test


class Logistic:
    def __init__(self, args):
        self.x_train, self.y_train, self.x_test, self.y_test = load_data(args.data)
        self.batch_size = args.batch_size
        # self.theta = [-1.0, 1.0, 1.0]
        self.theta = np.zeros((1, 3))
        self.bs = 0
        self.lr = args.lr
        self.epoch = args.epoch
        self.lr_decay = args.lr_decay
        self.is_decay = args.is_decay
        self.M = self.x_train.shape[1]
        self.N = self.x_train.shape[0]

    def normalization(self):
        # 均值方差归一化
        mean = np.mean(self.x_train)
        variance = np.std(self.x_train)
        self.x_train = (self.x_train - mean) / variance
        self.x_test = (self.x_train - mean) / variance
        self.x_train = np.insert(self.x_train, 0, values=1.0, axis=1)
        self.x_test = np.insert(self.x_train, 0, values=1.0, axis=1)
        self.y_train = self.y_train.reshape(self.N, 1)
        self.y_test = self.y_train.reshape(self.N, 1)
        self.M += 1

    def shuffle(self):
        # 实现的功能并非shuffle，而是随机挑选数据
        _x = np.zeros((self.batch_size, 3))
        _y = np.zeros(self.batch_size)
        if self.batch_size == -1:
            return self.x_train, self.y_train
        batch_size = self.batch_size
        max_index = int(self.y_train.shape[0]) - 1
        for i in range(batch_size):
            _ = random.randint(0, max_index)
            _x[i] = self.x_train[_]
            _y[i] = self.y_train[_]
        return _x, _y

    def gradient(self):
        l = 0
        g = np.zeros((1, 3))
        x, y = self.shuffle()
        for i in range(len(y)):
            _ = sigmoid(x[i] @ self.theta.T)
            g += (y[i] - _) * x[i]
            l += y[i] * math.log(_) + (1 - y[i]) * math.log(1 - _)
        return g, l

    def validate(self):
        theta = self.theta
        x_test = self.x_train
        y_test = self.y_train
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(y_test)):
            predict = 1 if float(x_test[i] @ self.theta.T) >= 0 else 0
            label = y_test[i]
            if predict == 0 and label == 0:
                TN += 1
            elif predict == 0 and label == 1:
                FN += 1
            elif predict == 1 and label == 0:
                FP += 1
            elif predict == 1 and label == 1:
                TP += 1
        precision = TP / (TP + FP+0.01)
        recall = TP / (TP + FN+0.01)
        F = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        return precision, recall, F, acc

    def train(self):
        # if self.batch_size != -1:
        # self.shuffle()
        self.normalization()
        plt.ion()
        plt.figure(figsize=(10, 4))
        precision, recall, F, acc = [], [], [], []
        _loss = []
        _i = []

        for i in range(self.epoch):
            _i.append(i)
            if i % 100 == 0 and self.is_decay:
                self.lr = self.lr * self.lr_decay
            g, l = self.gradient()
            self.theta += self.lr * g
            _loss.append(l)
            _precision, _recall, _F, _acc = self.validate()
            print("epoch:{} acc:{}".format(i, _acc))
            precision.append(_precision)
            recall.append(_recall)
            F.append(_F)
            acc.append(_acc)
            plt.cla()
            plt.subplot(1, 3, 1)
            plt.title("gd with lr decay" if self.is_decay else "gd without lr decay")
            plt.xlim((0, self.epoch))
            plt.ylim((-20, 0))
            plt.xlabel("epoch")
            plt.ylabel("likelihood")
            plt.plot(_i, _loss)
            # plt.figure(2)
            plt.subplot(1, 3, 2)
            plt.title("metrics")
            plt.xlim((0, self.epoch))
            plt.ylim((0, 1))
            plt.xlabel("epoch")
            plt.ylabel("per")
            l1, = plt.plot(_i, precision, color='green')
            l2, = plt.plot(_i, recall, color='blueviolet')
            l3, = plt.plot(_i, F, color='orangered')
            l4, = plt.plot(_i, acc, color='red', label='acc')
            plt.legend((l1, l2, l3, l4), ('precision', 'recall', 'F', 'acc'), loc='lower right', shadow=True)

            plt.subplot(1, 3, 3)
            plt.xlim((-2, 2))
            plt.ylim((-2, 2))
            for i in range(self.x_train.shape[0]):
                if self.y_train[i] == 0:
                    plt.plot(self.x_train[i][1], self.x_train[i][2], 'or')
                else:
                    plt.plot(self.x_train[i][1], self.x_train[i][2], 'ob')
            p1 = -(self.theta[0][0] + self.theta[0][1] * 2) / self.theta[0][2]
            p2 = -(self.theta[0][0] + self.theta[0][1] * -2) / self.theta[0][2]
            plt.plot([2, -2], [p1, p2])

            plt.pause(0.001)
            plt.clf()
        plt.ioff()
        plt.show()
        plt.pause(5)
        plt.close('all')

    def predict(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data', default='./data/Exam', help='data path')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='learning rate decay')
    parser.add_argument('--is_decay', default=True, type=bool, help='choose to use decay')
    args = parser.parse_args()
    model = Logistic(args)
    model.train()
