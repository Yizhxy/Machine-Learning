import numpy as np

N = 6


def predict(sentence, pai, BOW, theta, t, mood):
    ans = np.zeros(N)
    for i in range(N):
        p = 1 * pai[i]
        _cache = []
        for j in range(len(sentence)):
            word = sentence[j]
            index = BOW.index(word)
            if index not in _cache and mood == 2:
                _cache.append(index)
                continue
            p *= float(theta[i][index]) * t
        ans[i] = p
    _ = np.sum(ans)
    ans = ans / _
    # print(ans)
    c = np.argmax(ans)
    # print(Classes[c])
    return c
