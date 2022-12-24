# author: Bing-Cheng Chen
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import multivariate_normal
from numpy import genfromtxt
import matplotlib.pyplot as pyplot
from sklearn import metrics
import pickle

c = 8


def load_data(f):
    lb = []
    x1 = []
    x2 = []
    n = 0
    for line in f.readlines():
        if n == 0:
            n += 1
            continue
        else:
            line = line.strip('\n').split('\t')
            lb.append(int(line[0]))
            x1.append(float(line[1]))
            x2.append(float(line[2]))
    return {'lb': lb, 'x1': x1, 'x2': x2}, len(lb)


with open('results.pkl', 'rb') as f:
    [prior, mu, sigma, ll_evol] = pickle.load(f)
plt.ion()
# Show how the log-likelihood evolves as the training proceeds
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.set_title("Maximum likelihood ")
# pyplot.plot(ll_evol, 'o')
ax1.plot(ll_evol, 'o')
# pyplot.show()

# The learned mathematical expression for the GMM model after training on the given dataset
print('prior:', prior)
print('mu:', mu)
print('sigma:', sigma)

# Randomly select 500 data points from the given dataset and plot them on a 2-D coordinate system.
# Mark the data points coming from the same cluster with the same color.

# Reading the data file
# X = genfromtxt('t1.txt', delimiter=',')

file_path = "GMM" + str(c) + ".txt"
f = open(file_path)
gmm, n = load_data(f)
x1 = gmm['x1']
x2 = gmm['x2']
lbt = gmm['lb']
X = []
for i in range(n):
    X.append(np.asarray([x1[i], x2[i]]))
X = np.asarray(X)

print('data shape:', X.shape)

sel_num = len(X)
X_sel = []
sel_idxs = []
# while len(sel_idxs) < sel_num:
#     idx = np.random.randint(0, len(X), 1)
#     while idx in sel_idxs:
#         idx = np.random.randint(0, len(X), 1)
#     sel_idxs.append(idx[0])
# X_sel = X[sel_idxs]

X_sel = X


# get the labels of the points
def get_label(x, prior, mu, sigma):
    K = len(prior)
    p = np.zeros(K)
    for k in range(0, K):
        p[k] = prior[k] * multivariate_normal.pdf(x, mu[k, :], sigma[k, :, :])
    label = np.argmax(p)
    return label


lbs = []
for i in range(0, sel_num):
    lb = get_label(X_sel[i], prior, mu, sigma)
    lbs.append(lb)

# plot


ax2.set_title("predicted:" + str(c) + " categories")
ax2.scatter(X_sel[:, 0], X_sel[:, 1], marker='o', c=lbs)
# pyplot.scatter(X_sel[:,0], X_sel[:,1], marker='o', c=lbs)

ax3.set_title("true" + str(c) + " categories")
ax3.scatter(X[:, 0], X[:, 1], marker='o', c=lbt)

plt.ioff()
pyplot.show()

randindex = metrics.adjusted_rand_score(lbt, lbs)
lunkuoxishu = sklearn.metrics.silhouette_score(X, lbs, metric='euclidean', sample_size=None, random_state=None)
FMI = metrics.fowlkes_mallows_score(lbt, lbs)
print("Rand index:")
print(randindex)
print("Silhouette Coefficient:")
print(lunkuoxishu)
print("Fowlkes-Mallows scores")
print(FMI)
