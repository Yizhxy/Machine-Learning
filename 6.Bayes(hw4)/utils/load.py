from sklearn.feature_extraction.text import TfidfVectorizer
# from openpyxl import Workbook
import re

Classes = ['电脑', '法律', '教育', '经济', '体育', '政治']
# Classes = ['电脑']
N = len(Classes)


def stopWord(path):
    stopwords = []
    with open(path + "./stop_words_zh.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            line = i.strip()
            stopwords.append(line)
    stopwords = list(set(stopwords))
    return stopwords


def load_data(path, data, label):
    res = 0
    for i in range(N):
        with open("Tsinghua./" + path + './' + Classes[i] + ".txt", 'r', encoding='utf-8') as f:
            lines = f.read()
            dtext = re.findall(r'<text>\n(.*?)\n</text>', lines)

        for tmp in dtext:
            res += 1
            seg_list = tmp.split()
            seg_str = ",".join(seg_list)
            data.append(seg_str)
            label.append(i)
    return res
