import numpy as np

Classes = ['电脑', '法律', '教育', '经济', '体育', '政治']
N = len(Classes)


def load(path):
    data_train = []
    data_test = []
    for i in range(N):
        _data = []
        _path_train = path + './train./' + Classes[i] + '.txt'
        _path_test = path + './train./' + Classes[i] + '.txt'
        with open(_path_train, 'r', encoding="utf-8") as f:
            txt = f.read()
            txt = txt.split('<text>')
        for i in range(1, len(txt)):
            a = txt[i].split('</text>')[0].split('\n')[1]
            _data.append(a)
        data_train.append(_data)
        _data = []
        with open(_path_test, 'r', encoding="utf-8") as f:
            txt = f.read()
            txt = txt.split('<text>')
        for i in range(1, len(txt)):
            a = txt[i].split('</text>')[0].split('\n')[1]
            _data.append(a)
        data_test.append(_data)
    return data_train, data_test
