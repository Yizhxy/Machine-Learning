import numpy as np
import matplotlib.pyplot as plt

import argparse
import math
import random
import matplotlib as mpl


def load_data(path):
    X_train = np.loadtxt(path + "/train/x.txt")
    Y_train = np.loadtxt(path + "/train/y.txt", dtype=int)
    X_test = np.loadtxt(path + "/test/x.txt")
    Y_test = np.loadtxt(path + "/test/y.txt", dtype=int)
    return X_train, Y_train, X_test, Y_test


def cross_entropy(y_hat, t, c):
    '''
    :param y_hat: 预测分布
    :param t: 真实标签 为一个标量
    :param c: 类别数目
    :return: 交叉熵
    '''
    _y = np.zeros((y_hat.shape[0], c))
    for i in range(y_hat.shape[0]):
        _ = int(t[i][0])
        _y[i][_] = 1
    ans = np.zeros((y_hat.shape[0], 1))
    for i in range(y_hat.shape[0]):
        ans[i] = -np.sum(_y * np.log(y_hat))
    return np.mean(ans)


class Linear:
    def __init__(self, input_dim, output_dim, L2=0., lr=0.01):
        '''
        :param L2:L2正则化系数
        '''
        self.x = None
        self.theta = np.zeros((input_dim, output_dim))
        self.bias = np.ones((1, output_dim))
        self.n = (input_dim + 1) * output_dim
        self.lr = lr
        self.L2 = L2

    def forward(self, x):
        # param x: [B,input_dim]
        self.x = x
        return x @ self.theta + self.bias

    def weights(self):
        return self.L2 / 2 * (np.sum(self.theta ** 2) + np.sum(self.theta ** 2)) / self.n

    def bp(self, gradient):
        # param loss:[B,output_dim]
        self.theta = self.theta * (1 - self.lr * self.L2 / self.n) - self.lr * (self.x.T @ gradient)
        self.bias = self.bias * (1 - self.lr * self.L2 / self.n) - self.lr * np.ones((1, gradient.shape[0])) @ gradient
        return gradient @ self.theta.T


class Softmax:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        _ = np.zeros(x.shape)
        for i in range(_.shape[0]):
            _[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
        return _

    def bp(self, gradient):
        return gradient


class Sigmoid:
    def __init__(self):
        self.z = None

    def forward(self, x):
        # param x:[B,input_dim]
        self.z = 1 / (1 + np.exp(-x))
        return self.z

    def bp(self, gradient):
        _ = self.z * (1 - self.z)
        return gradient * _


class Tanh:
    def __init__(self):
        self.z = None
        self.net = Sigmoid()

    def forward(self, x):
        self.z = 2 * self.net.forward(2 * x) - 1
        return 2 * self.net.forward(2 * x) - 1

    def bp(self, gradient):
        return gradient * (1 - self.z ** 2)


class LeakyRelu:
    def __init__(self, a=0.1):
        # a=0时为Relu
        self.m = None
        self.a = a
        pass

    def forward(self, x):
        self.m = (x <= 0)
        out = x.copy()
        out[self.m] = self.a * out[self.m]
        return out

    def bp(self, gradient):
        gradient[self.m] = 0
        return gradient


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
        creat_model.append(Linear(self.M, 2 * self.M - 1, L2=0.1))
        creat_model.append(Sigmoid())
        creat_model.append(Linear(2 * self.M - 1, self.L, L2=0.1))
        creat_model.append(Softmax())
        return creat_model

    def normalization(self):
        mean = np.mean(self.X_train)
        variance = np.std(self.X_train)
        self.X_train = (self.X_train - mean) / variance
        self.X_test = (self.X_test - mean) / variance
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

    def predict(self, x):
        _ = self.forward(x)
        ans = np.zeros((_.shape[0], 1))
        for i in range(_.shape[0]):
            ans[i] = np.argmax(_[i])
        return ans

    def bp(self, x, y):
        _y = np.zeros((self.batch_size, self.L))
        for i in range(self.batch_size):
            _ = int(y[i][0])
            _y[i][_] = 1
        _ = x - _y
        for i in range(len(self.model) - 1, -1, -1):
            _ = self.model[i].bp(_)

    def eval(self, is_training=False):
        if not is_training:
            x = self.X_test
            y = self.Y_test
        else:
            x = self.X_train
            y = self.Y_train
        T = 0
        y_hat = self.forward(x)
        for i in range(x.shape[0]):
            predict = np.argmax(y_hat[i])
            if predict == y[i]:
                T += 1
        # print(T / x.shape[0])
        return T / x.shape[0]

    def train(self):
        x, y = self.shuffle()
        _ = self.forward(x)
        loss = cross_entropy(_, y, self.L)
        for i in range(len(self.model)):
            if isinstance(self.model[i], Linear):
                loss += self.model[i].weights()
        self.bp(_, y)
        return loss

    def main(self):
        plt.ion()
        plt.figure(figsize=(10, 3))
        _acc_train, _acc_test, _i = [], [], []
        _loss = []
        for i in range(self.epoch):
            _loss.append(self.train())
            acc_train = self.eval(is_training=True)
            acc_test = self.eval()
            _i.append(i)
            _acc_train.append(acc_train)
            _acc_test.append(acc_test)
            plt.cla()

            plt.subplot(1, 3, 1)
            plt.title("gd")
            plt.xlim((0, self.epoch))
            plt.ylim((0, 50))
            plt.xlabel("epoch")
            plt.ylabel("avg loss")
            loss_line, = plt.plot(_i, _loss, 'b')
            plt.legend((loss_line,), ('train',), loc='upper right', fontsize='small')

            plt.subplot(1, 3, 2)
            plt.xlim((0, self.epoch))
            plt.ylim((0, 1))
            plt.xlabel("epoch")
            plt.ylabel("acc")
            line1, = plt.plot(_i, _acc_train, 'r')
            line2, = plt.plot(_i, _acc_test, 'b')
            plt.legend((line1, line2), ('train', 'test'), loc='lower right', fontsize='small')

            plt.subplot(1, 3, 3)
            X = self.X_train
            plot_step = 0.10
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))  # 生成网格采样点
            grid_test = np.stack((xx.flat, yy.flat), axis=1)  # 生成测试点
            grid_hat = self.predict(grid_test)  # 用训练好的模型对测试点进行预测
            grid_hat = grid_hat.reshape(xx.shape)
            cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 由于3分类，设置3种颜色或者多种
            plt.pcolormesh(xx, yy, grid_hat, cmap=cm_light)  # 此处的xx,yy必须是网格采样点
            # plt.pcolormesh()会根据grid_hat的结果自动在cmap里选择颜色,作用在于能够直观表现出分类边界
            cm_dark = mpl.colors.ListedColormap(['g', 'r', 'y'])
            plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.Y_train, edgecolors='k', s=50,
                        cmap=cm_dark)  # 用散点图把样本点画上去
            plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.Y_test, edgecolors='k', s=50, cmap=cm_dark)

            plt.pause(0.001)
            plt.clf()
        plt.ioff()
        plt.show()
        plt.pause(5)
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data_path', default='./data/Iris', help='data path')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--epoch', default=1000, type=int, help='epoch')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='learning rate decay')
    parser.add_argument('--is_decay', default=True, type=bool, help='choose to use decay')
    args = parser.parse_args()
    X_train, Y_train, X_test, Y_test = load_data(args.data_path)
    model = FNN(args, X_train, Y_train, X_test, Y_test, epoch=args.epoch, lr=args.lr, is_decay=args.is_decay,
                lr_decay=args.lr_decay,
                batch_size=args.batch_size)
    model.main()
