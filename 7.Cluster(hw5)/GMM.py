import argparse

from numpy import *
from numpy import max as Max
from numpy import min as Min
import matplotlib.pyplot as plt
from random import sample as Sample
from sklearn.model_selection import KFold
import numpy as np

parser = argparse.ArgumentParser(
    description=__doc__)
parser.add_argument('--data_path', default='GMM3', type=str, help='data')
args = parser.parse_args()
path = args.data_path

# 是否采用随机梯度下降法
is_random = 0

# 随机梯度下降中一次取得的样本数
batch_size = 100

# 迭代次数
iterate_num = 500

# 收敛阈值
lmt = 1e-6

# 点的大小
dot_size = 5


# 读取文件


# 读取数据
def load_data(path):
    path = str('gmm./' + path + '.txt')
    data = np.loadtxt(path)
    data = data.T
    x = data[1:].T
    y = data[0]
    y = np.array(y, dtype=int)
    return x, y


# 最大-最小归一化处理
def max_min_normalization(x):
    # 一定要用numpy里的max和min，不然很joker
    x = (x - Min(x)) / (Max(x) - Min(x))
    return array(x)


# n为样本数量
gmm, lb = load_data(path)
nn = gmm.shape[0]
c = np.max(lb) + 1
# 对数据进行最大-最小归一化处理
xx1 = max_min_normalization(gmm[:, 0])
xx2 = max_min_normalization(gmm[:, 1])

yy = array([lb[k] for k in range(nn)])

# 5折交叉检验
avg_acc = 0
test = 1
if test == 1:
    print("test:")
else:
    print("train:")
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(yy):
    # 学习率
    alpha = 1e-2
    # 准确率计算
    acc = 0
    # x是存放n个样本特征向量[1.0,x1,x2]的列表，x[k]表示第k个样本的特征向量
    # y是存放n个样本对应的真实标签，y[k]表示第k个样本的真实标签
    x1, x2, y = xx1[train_index], xx2[train_index], yy[train_index]
    test_x1, test_x2, test_y = xx1[test_index], xx2[test_index], yy[test_index]
    n = len(x1)
    test_n = len(test_x1)
    x = [mat([x1[k], x2[k]]).transpose() for k in range(n)]
    test_x = [mat([test_x1[k], test_x2[k]]).transpose() for k in range(test_n)]

    pij = []
    u = []
    ep = []
    for j in range(c):
        sum = 0
        sum_up1 = zeros((2, 1))
        sum_up2 = zeros((2, 2))
        for k in range(n):
            if y[k] == j:
                sum += 1
                sum_up1 = sum_up1 + x[k]
        uj = sum_up1 / sum
        for k in range(n):
            if y[k] == j:
                sum_up2 = sum_up2 + matmul(x[k] - uj, (x[k] - uj).transpose())
        epj = sum_up2 / sum
        pij.append(sum / n)
        u.append(uj)
        ep.append(epj)


    # print(pij)
    # print(u)
    # print(ep)
    def pre(x, j):
        mul1 = matmul((x - u[j]).transpose(), linalg.inv(ep[j]))
        mul2 = matmul(mul1, x - u[j])
        exp_mul = exp(-1 / 2 * mul2)
        det = 2 * pi * sqrt(linalg.det(ep[j]))
        return pij[j] * exp_mul / det


    def pre_label(x):
        pre_list = [pre(x, j) for j in range(c)]
        return argmax(pre_list)


    figure = plt.figure(figsize=(9.5, 2.8))
    ax0 = figure.add_subplot(1, 3, 1)
    ax1 = figure.add_subplot(1, 3, 2)
    ax2 = figure.add_subplot(1, 3, 3)
    ax0.scatter(xx1, xx2, c=yy, s=dot_size)
    ax0.set_title("ALL DATA")
    ax1.set_title("predicted test set")
    ax2.set_title("true test set")

    if test == 1:
        prel = []
        for k in range(test_n):
            prel.append(pre_label(test_x[k]))
            if pre_label(test_x[k]) == test_y[k]:
                acc += 1
        acc /= test_n
        print("acc=", acc)
        testx = np.array(test_x)
        ax1.scatter(testx[:, 0], testx[:, 1], c=prel, s=dot_size)
        ax2.scatter(testx[:, 0], testx[:, 1], c=test_y, s=dot_size)
        plt.show()

    else:
        for k in range(n):
            if pre_label(x[k]) == y[k]:
                acc += 1
        acc /= n
        print("acc=", acc)
    avg_acc += acc
avg_acc /= 5
print("avg_acc=", avg_acc)
