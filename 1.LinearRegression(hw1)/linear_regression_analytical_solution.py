import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_data(path):
    data = path
    with open(data + "/x.txt", "r") as f:
        _ = f.readlines()
        x = []
        for i in range(len(_)):
            x.append(float(_[i].split('\n')[0]))
            x.append(1.)
    with open(data + "/y.txt", "r") as f:
        _ = f.readlines()
        y = []
        for i in range(len(_)):
            y.append(float(_[i].split('\n')[0]))
    return x, y


def main(args):
    data = args.data
    x, y = get_data(data)
    _x = [x[i] for i in range(0, len(x), 2)]
    _y = [y[i] for i in range(0, len(y))]
    assert len(x) // 2 == len(y), "the shape of x and y are inconsistent "
    x = np.array(x).reshape(len(x) // 2, 2)
    x, y = np.matrix(x), np.matrix(y).T
    _ = (x.T @ x)
    theta = np.linalg.inv(_) @ x.T @ y
    y_predict_2000 = float(theta[0] * 2000 + theta[1])
    y_predict_2020 = float(theta[0] * 2020 + theta[1])
    year = args.predict_year
    year_predict = float(theta[0] * year + theta[1])
    y_predict = [y_predict_2000, y_predict_2020]
    plt.plot(_x, _y, 'o')
    plt.plot(year, year_predict, 'or')
    plt.plot([2000, 2020], y_predict)
    plt.title("AS:the price forecast for {} is {}".format(year, round(year_predict, 3)))
    loss = 1 / 2 * (x @ theta - y).T @ (x @ theta - y)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data', default='./data/Price', help='data path')
    parser.add_argument('--predict_year', default=2014, type=int, help='the year to predict')
    args = parser.parse_args()
    main(args)
