import argparse

import sklearn
from numpy import *
from numpy import max as Max
from numpy import min as Min
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans


def load_data(path):
    path=str('gmm./'+path+'.txt')
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
    return x


# plt.scatter(x1, x2, c="gray")
# plt.show()

def distance(triple1, triple2):
    return (triple1[0] - triple2[0]) ** 2 + (triple1[1] - triple2[1]) ** 2


def main(path):
    # 迭代次数
    iterate_num = 50
    # 学习率
    alpha = 1e-2
    # 收敛值
    lmt = 1e-5
    # 点的大小
    dot_size = 5
    # 读取文件
    gmm, lb = load_data(path)
    gmm[:, 0] = max_min_normalization(gmm[:, 0])
    gmm[:, 1] = max_min_normalization(gmm[:, 1])
    x1 = gmm[:, 0]
    x2 = gmm[:, 1]
    n = gmm.shape[0]
    c = np.max(lb)+1
    x = [{'x1': gmm[k][0], 'x2': gmm[k][1]} for k in range(n)]
    y = lb
    # 随机选取样本点
    m = [(random.choice(x)['x1'], random.choice(x)['x2']) for k in range(c)]
    w = [zeros((3, 1)) for j in range(c)]
    # 样本点类别列表
    cc = [{'x1': [], 'x2': []} for k in range(c)]

    # 绘制子图
    figure = plt.figure(figsize=(9.5, 2.8))
    ax0 = figure.add_subplot(1, 3, 1)
    ax1 = figure.add_subplot(1, 3, 2)
    ax2 = figure.add_subplot(1, 3, 3)
    ax0.scatter(x1, x2, c=y, s=dot_size)
    ax0.set_title("Train Data")
    ax2.set_title("WCSS")

    wcss_list = []
    labels = []
    plt.ion()
    for tot in range(iterate_num):
        labels.clear()
        for j in range(c):
            cc[j]['x1'].clear()
            cc[j]['x2'].clear()
        ax1.set_title("K-means")
        ax2.set_xlim(0, iterate_num)
        wcss = 0
        for i in range(n):
            # 得到样本点与各个类中心点的距离，并存储在列表中
            dis = [distance((x1[i], x2[i]), m[k]) for k in range(c)]

            # 根据欧氏距离划分样本集，对其进行分类并定标签
            # 标签即为列表dis的最小元素的下标k=1,...,c
            label = argmin(dis)

            # 累加平方欧式距离
            wcss += min(dis)

            # 对样本点进行分配
            # 处理成这样是为了方便求x1和x2的均值
            cc[label]['x1'].append(x1[i])
            cc[label]['x2'].append(x2[i])

            # 颜色列表
            labels.append(label)

        wcss_list.append(wcss)
        ax1.scatter(x1, x2, c=labels, s=dot_size)
        # 绘制类中心点
        for k in range(c):
            ax1.scatter(m[k][0], m[k][1], c="black", marker="x")
        plt.pause(0.1)
        ax1.cla()
        # print(wcss)
        # 更新
        # 体现了方便之处
        m = [(mean(cc[k]['x1']), mean(cc[k]['x2'])) for k in range(c)]
        if tot > 0:
            ax2.plot([tot, tot + 1], [wcss_list[tot - 1], wcss_list[tot]], c='black')
            if abs(wcss_list[tot] - wcss_list[tot - 1]) < lmt:
                break
    # 最终的分类结构可视化
    ax1.scatter(x1, x2, c=labels, s=dot_size)
    ax1.set_title("K-means")
    plt.ioff()
    plt.pause(0.7)
    plt.show()

    randindex = metrics.adjusted_rand_score(lb, labels)
    lunkuoxishu = sklearn.metrics.silhouette_score(gmm, labels, metric='euclidean', sample_size=None, random_state=None)
    FMI = metrics.fowlkes_mallows_score(lb, labels)

    print("Rand index:")
    print(randindex)
    print("Silhouette Coefficient:")
    print(lunkuoxishu)
    print("Fowlkes-Mallows scores")
    print(FMI)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data_path', default='GMM8', type=str, help='data')
    args = parser.parse_args()
    main(args.data_path)
