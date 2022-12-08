import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def load_data(path):
    X_train = np.loadtxt(path + "/train/x.txt")
    Y_train = np.loadtxt(path + "/train/y.txt", dtype=int)
    X_test = np.loadtxt(path + "/test/x.txt")
    Y_test = np.loadtxt(path + "/test/y.txt", dtype=int)
    return X_train, Y_train, X_test, Y_test


class TrainSet(Dataset):
    def __init__(self, path):
        self.X_train = np.loadtxt(path + "/train/x.txt")
        self.Y_train = np.loadtxt(path + "/train/y.txt", dtype=int)
        _y = np.zeros((self.Y_train.shape[0], max(self.Y_train) + 1))
        self.normalization()
        for i in range(self.Y_train.shape[0]):
            _ = int(self.Y_train[i])
            _y[i][_] = 1
        self.Y_train = _y
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.float32)
        # self.X_train = torch.from_numpy(self.X_train)
        # self.Y_train = torch.from_numpy(self.Y_train)

    def get_data(self):
        return self.X_train, self.Y_train

    def getL(self):
        return self.Y_train.shape[1]

    def normalization(self):
        # 均值方差归一化
        mean = np.mean(self.X_train)
        variance = np.std(self.X_train)
        self.X_train = (self.X_train - mean) / variance

    def __getitem__(self, item):
        return self.X_train[item], self.Y_train[item]

    def __len__(self):
        return len(self.X_train)


class FNN(nn.Module):
    def __init__(self, opt):
        super(FNN, self).__init__()
        self.epoch = opt.epoch
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.dataset = TrainSet(opt.data_path)
        self.x, self.y = self.dataset.get_data()
        self.L = self.dataset.getL()
        self.train_loader = DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True)
        self.model = self.init_model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr)

    def init_model(self):
        creat_model = nn.Sequential(
            nn.Linear(2, 10, bias=True),
            nn.Sigmoid(),
            nn.Linear(10, self.L, bias=True),
            nn.Softmax(dim=-1),
        )
        return creat_model

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def get_acc(self, x, y):
        x = self.x
        y = self.y
        T = 0
        y_hat = self.model.forward(x)
        for i in range(x.shape[0]):
            predict = torch.argmax(y_hat[i])
            if predict == torch.argmax(y[i]):
                T += 1
        # print(T / x.shape[0])
        return T / x.shape[0]

    def main(self):
        self.model.train()
        for i in range(self.epoch):
            for j, (x, y) in enumerate(self.train_loader):
                y_hat = self.forward(x)
                loss = self.loss_func(y_hat, y)
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = self.get_acc(x, y)
                print("[{} {}] loss: {} acc:{}".format(i, j, float(loss), acc))


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
    fnn = FNN(args)
    fnn.main()
