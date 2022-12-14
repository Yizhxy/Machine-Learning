def word2index(BOW):
    word2index_map = {}
    for i in range(len(BOW)):
        word = BOW[i]
        word2index_map.setdefault(word, i)
    return word2index_map
