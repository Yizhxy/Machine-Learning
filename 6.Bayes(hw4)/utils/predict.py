import numpy as np

N = 6


def predict(sentence, pai, BOW, theta, t, mood, stop_words, w2i):
    ans = np.zeros(N)
    for i in range(N):
        p = 1 * pai[i]
        _cache = []
        for j in range(len(sentence)):
            word = sentence[j]
            if word in stop_words:
                continue
            index = w2i[word]
            if index not in _cache and mood == 2:
                _cache.append(index)
                p *= float(theta[i][index]) * t
                continue
            elif mood == 1:
                p *= float(theta[i][index]) * t
        ans[i] = p
    _ = np.sum(ans)
    # ans = ans / _
    # print(ans)
    c = np.argmax(ans)
    # print(Classes[c])
    return c
