import numpy as np
from .load import load_data
from .load import stopWord


def init():
    num = 0
    data = []
    label = []
    num += load_data('train', data, label)
    _ = load_data('test', data, label)
    stop_words = stopWord('Tsinghua')
    BOW = []
    pai = np.zeros(6)
    for i in range(6):
        pai[i] = label[: num].count(i)
    pai = pai / np.sum(pai)
    for i in range(len(data)):
        print("加载数据中:" + str(i / len(data)))
        data[i] = data[i].split(',')
        for j in range(len(data[i])):
            if data[i][j] not in BOW and data[i][j] not in stop_words:
                BOW.append(data[i][j])
    return BOW, pai, data, label, num, stop_words
