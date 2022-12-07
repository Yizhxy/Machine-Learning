# coding=UTF-8
import random
import matplotlib.pyplot as plt
import numpy as np


def getdata(filename):
    x = []
    y = []
    with open(filename) as file:
        for line in file:
            lineArr = line.strip().split('\t')
            x.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
            y.append(float(lineArr[2]))  # 添加标签
    return x, y


def drawdata(x, y):
    plus = []
    minus = []
    for i in range(len(x)):
        if y[i] > 0:
            plus.append(x[i])
        else:
            minus.append(x[i])
    plus_matrix = np.array(plus)
    minus_matrix = np.array(minus)
    plt.scatter(np.transpose(plus_matrix)[0], np.transpose(plus_matrix)[1])
    plt.scatter(np.transpose(minus_matrix)[0], np.transpose(minus_matrix)[1])
    plt.show()


class SVM(object):
    """
    参数：
        x: 数据
        y: 标签
        c: 松弛变量
        toler: 容错率
        n_iter: 最大迭代次数
    """

    def __init__(self, c=0.6, toler=0.001, n_iter=40):
        self.c = c
        self.toler = toler
        self.n_iter = n_iter
        self.alphas = np.array([])
        self.w_ = []
        self.b = 0

    def fit(self, x, y):
        # 转换为矩阵
        x_matrix = np.mat(x)
        y_matrix = np.mat(y).transpose()  # 矩阵转置
        # 利用SMO计算b和alpha
        self.b, self.alphas = self.smosimple(x_matrix, y_matrix, self.c, self.toler)
        self.w_ = np.zeros((x_matrix.shape[1], 1))
        for i in range(self.alphas.shape[0]):
            self.w_ += np.multiply(self.alphas[i] * y_matrix[i], x_matrix[i, :].T)
        return self

    # 简化版SMO
    def smosimple(self, x_matrix, y_matrix, C, toler):
        # 初始化b参数
        b = 0
        # 统计x矩阵维度(m行n列)
        m, n = np.shape(x_matrix)
        # 初始化alpha参数
        alphas = np.mat(np.zeros((m, 1)))
        count = 0
        # alphas为矩阵，y为矩阵，x为矩阵，
        while count < self.n_iter:
            alphaschanged = 0
            for i in range(m):
                # 步骤1：计算误差E
                # Ei = (sum[aj * yj * K(xi,xj)] + b) - yi；误差 = 预测值 - 真实值
                fi = float(np.multiply(alphas, y_matrix).T * self.kernel(x_matrix[i, :], x_matrix)) + b
                Ei = fi - float(y_matrix[i])

                # 优化alpha
                # 满足KKT条件
                if ((y_matrix[i] * Ei < -toler) and (alphas[i] < C)) or (
                        (y_matrix[i] * Ei > toler) and (alphas[i] > 0)):
                    # 随机选择另一个与 alpha_i 成对优化的 alpha_j
                    j = self.selectJrand(i, m)
                    fj = float(np.multiply(alphas, y_matrix).T * self.kernel(x_matrix[j, :], x_matrix)) + b
                    Ej = fj - float(y_matrix[j])
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()  # 深拷贝，不随原数据修改而修改
                    # 步骤2：计算上下界L和H
                    if y_matrix[i] != y_matrix[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    if L == H:
                        continue
                    # 步骤3：计算学习率
                    # eta = K11 + K22 - 2*K12
                    eta = (self.kernel(x_matrix[i, :], x_matrix[i, :])
                           + self.kernel(x_matrix[j, :], x_matrix[j, :])
                           - 2.0 * self.kernel(x_matrix[i, :], x_matrix[j, :]))
                    if eta <= 0:
                        continue
                    # 步骤4：更新 alpha_j
                    alphas[j] += y_matrix[j] * (Ei - Ej) / eta
                    # 步骤5：对alpha_j进行剪枝
                    alphas[j] = self.clipper(alphas[j], H, L)
                    # 步骤6：更新alpha_i
                    alphas[i] += y_matrix[i] * y_matrix[j] * (alphaJold - alphas[j])
                    # 步骤7：更新b1,b2,b
                    b1 = (- Ei
                          - y_matrix[i] * self.kernel(x_matrix[i, :], x_matrix[i, :]) * (alphas[i] - alphaIold)
                          - y_matrix[j] * self.kernel(x_matrix[j, :], x_matrix[i, :]) * (alphas[j] - alphaJold)
                          + b)
                    b2 = (- Ej
                          - y_matrix[i] * self.kernel(x_matrix[i, :], x_matrix[j, :]) * (alphas[i] - alphaIold)
                          - y_matrix[j] * self.kernel(x_matrix[j, :], x_matrix[j, :]) * (alphas[j] - alphaJold)
                          + b)
                    if (0 < alphas[i]) and (C > alphas[i]):
                        b = b1
                    elif (0 < alphas[j]) and (C > alphas[j]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaschanged += 1
            if alphaschanged == 0:
                count += 1
        return b, alphas

    def kernel(self, xi, xj):
        return xj * xi.T

    def selectJrand(self, i, m):
        while True:
            j = int(random.uniform(0, m))
            if j != i:
                return j

    def clipper(self, alphas, H, L):
        if alphas > H:
            return H
        elif L <= alphas <= H:
            return alphas
        elif alphas < L:
            return L

    def drawresult(self, x, y):
        # x = np.mat(x)
        # y = np.mat(y).transpose()
        plus = []
        minus = []
        for i in range(len(x)):
            if y[i] > 0:
                plus.append(x[i])
            else:
                minus.append(x[i])
        plus_matrix = np.array(plus)
        minus_matrix = np.array(minus)
        plt.scatter(np.transpose(plus_matrix)[0], np.transpose(plus_matrix)[1], s=30, alpha=0.7)
        plt.scatter(np.transpose(minus_matrix)[0], np.transpose(minus_matrix)[1], s=30, alpha=0.7)

        x1 = max(x)[0]
        x2 = min(x)[0]
        a1, a2 = self.w_
        b = float(self.b)
        a1 = float(a1[0])
        a2 = float(a2[0])
        y1 = (-b - a1 * x1) / a2
        y2 = (-b - a1 * x2) / a2
        plt.plot([x1, x2], [y1, y2])

        for i, alpha in enumerate(self.alphas):
            if abs(alpha) > 0:
                X, Y = x[i]
                plt.scatter([X], [Y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
        plt.show()


if __name__ == '__main__':
    filename = "./testSet.txt"
    x, y = getdata(filename)
    # drawdata(x, y)
    svm = SVM()
    svm.fit(x, y)
    # print(svm.w_)
    svm.drawresult(x, y)

