import numpy as np
from .predict import predict

N = 6


def eval(data, pai, BOW, theta, label, stop_words,w2i, t, mood, ):
    T = 0
    K = len(data)
    for i in (range(len(data))):
        if predict(data[i], pai, BOW, theta, t, mood, stop_words,w2i) == label[i]:
            T += 1
    acc = T / K
    return acc
