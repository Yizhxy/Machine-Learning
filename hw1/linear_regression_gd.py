import numpy as np
import argparse
import matplotlib.pyplot as plt


def get_data(path):
    data = path
    with open(data + "/x.txt", "r") as f:
        _ = f.readlines()
        _x = []
        for i in range(len(_)):
            _x.append(float(_[i].split('\n')[0]))
    with open(data + "/y.txt", "r") as f:
        _ = f.readlines()
        y = []
        for i in range(len(_)):
            y.append(float(_[i].split('\n')[0]))
    assert len(_x) == len(y), "the shape of x and y are inconsistent "
    x = np.zeros((len(_x), 2), dtype=float)
    x_min = min(_x)
    x_max = max(_x)
    x[:, 0] = (np.array(_x) - x_min) / (x_max - x_min)
    x[:, 1] = np.ones(len(_x))
    return x, y, _x, x_min, x_max


class LR:
    def __init__(self, args):
        self.x, self.y, self._x, self.min, self.max = get_data(args.data)
        self._y = [self.y[i] for i in range(0, len(self.y))]
        self.x, self.y = np.matrix(self.x), np.matrix(self.y).T
        self.theta = np.matrix(np.array([[15.], [2.1]]))
        self.epoch = args.epoch
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.is_decay = args.is_decay
        self.year = args.predict_year

    def gradient(self):
        x = self.x
        y = self.y
        theta = self.theta
        return x.T @ x @ theta - x.T @ y

    def loss(self):
        x = self.x
        y = self.y
        theta = self.theta
        loss = 1 / 2 * (x @ theta - y).T @ (x @ theta - y)
        return loss

    def train(self):
        # plt.ion()
        _i = []
        _loss = []
        for i in range(self.epoch):
            if i % 100 == 0 and self.is_decay:
                self.lr = self.lr_decay * self.lr
            loss = self.loss()
            print("epoch:{},loss:{}".format(i, round(float(loss), 3)))
            g = self.gradient()
            self.theta[0] -= self.lr * g[0]
            self.theta[1] -= self.lr * g[1]
            _i.append(i)
            _loss.append(float(loss))
            '''用以绘制loss图
            plt.cla()
            plt.title("gd with lr decay" if self.is_decay else "gd without lr decay")
            plt.xlim((0, 200))
            plt.ylim((0, 100))
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(_i, _loss)
            plt.pause(0.0001)
        plt.ioff()
        plt.show()'''
        '''用以调试查看最终loss及参数
        x = self.x
        y = self.y
        theta = self.theta
        loss = 1 / 2 * (x @ theta - y).T @ (x @ theta - y)'''

    def predict(self):
        _x = self._x
        _y = self._y
        x = [(2000 - self.min) / (self.max - self.min), (2020 - self.min) / (self.max - self.min)]
        y_predict = [float(self.theta[0] * x[i] + self.theta[1]) for i in range(len(x))]
        year_predict = self.theta[0] * (self.year - self.min) / (self.max - self.min) + self.theta[1]
        plt.plot(_x, _y, 'o')
        plt.plot(self.year, year_predict, 'or')
        plt.plot([2000, 2020], y_predict)
        plt.title("GD:the price forecast for {} is {}".format(self.year, round(float(year_predict), 3)))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data', default='./data/Price', help='data path')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5000, type=int, help='epoch')
    parser.add_argument('--predict_year', default=2014, type=int, help='the year to predict')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='learning rate decay')
    parser.add_argument('--is_decay', default=True, type=bool, help='choose to use decay')
    args = parser.parse_args()
    lr_model = LR(args)
    lr_model.train()
    lr_model.predict()
