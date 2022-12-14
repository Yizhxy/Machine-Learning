import numpy as np
from .predict import predict

N = 6


def eval(data, pai, BOW, theta, t, mood):
    T = 0
    K = 0
    for i in range(N):
        K += len(data[i])
        for j in range(len(data[i])):
            if predict(data[i][j], pai, BOW, theta, t, mood) == i:
                T += 1
    acc = T / K
    return acc
