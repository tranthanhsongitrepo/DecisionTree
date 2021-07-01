from collections import deque

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from matplotlib import pyplot as plt


class Node:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_children(self, children):
        self.children.append(children)

    def __str__(self):
        res = ""
        for data in self.data:
            res += str(data) + " "
        return res


class DecisionTree:
    def __init__(self):
        self.root = Node([None, -1, -1])

    def walk(self, x):
        res = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            res[i] = int(self.walk_util(self.root, x[i, :]))
        return res

    def walk_util(self, node, s):
        # Leaf node

        if len(node.children) == 0:
            return node.data[2]

        label_occurrence = {}
        for child in node.children:
            label_occurrence[child.data[2]] = 0
            if child.data[1] == s[node.data[0]]:
                return self.walk_util(child, s)

        max_label = 0
        max_label_occurrence = 0
        for child in node.children:
            label_occurrence[child.data[2]] += 1

            if label_occurrence[child.data[2]] > max_label_occurrence:
                max_label = child.data[2]
                max_label_occurrence = label_occurrence[child.data[2]]
        return max_label


class ID3(BaseEstimator, ClassifierMixin):
    def __init__(self, thresh=None, max_depth=None):
        super().__init__()
        # self.c = y.unique()
        self._tree = DecisionTree()
        self._q = deque()
        self._tree_q = deque()
        if thresh is not None:
            self.thresh = thresh
        else:
            self.thresh = None
        self.max_depth = max_depth

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def entropy(self, x):
        n = x.shape[0]
        _, t = np.unique(x, return_counts=True, axis=0)
        t = t / n
        t = -t * np.log(t)
        return t.sum()

    def h(self, x):
        h_min = 2
        h_min_index = -1
        min_child = np.array([])

        if np.unique(x[:, -1]).shape[0] != 1:
            for i in range(x.shape[1] - 1):
                col = x[:, i]
                uniques, uniques_count = np.unique(col, return_counts=True)
                if uniques.shape[0] == 1:
                    continue

                h = 0
                for unique in uniques:
                    mask = col == unique
                    t = col[mask].reshape((-1, 1))
                    y = x[:, -1][mask].reshape((-1, 1))
                    h += t.shape[0] * self.entropy(np.hstack((t, y))) / x.shape[0]
                if h < h_min:
                    h_min = h
                    h_min_index = i
                    min_child = uniques

        # Prevent overfitting by stopping early
        if self.thresh is not None and h_min > 1 - self.thresh:
            return 0, -1, np.array([])
        return h_min, h_min_index, min_child

    def fit(self, X, y):
        cur_depth = 0
        c = 1
        data = np.hstack((X, y.reshape((-1, 1))))
        self._q.append(data)
        self._tree_q.append(self._tree.root)
        while len(self._q) > 0:
            c -= 1
            top = self._q.popleft()
            h_min, h_min_index, min_child = self.h(top)
            prev = self._tree_q.popleft()
            prev.data[0] = h_min_index
            y_uni, y_uni_count = np.unique(top[:, -1], return_counts=True)
            prev.data[2] = y_uni[y_uni_count.argmax()]
            if self.max_depth is None or (
                    self.max_depth is not None and cur_depth < self.max_depth):
                for child in min_child:
                    node = Node([None, child, -1])
                    mask = top[:, h_min_index] == child

                    self._q.append(top[mask])
                    self._tree_q.append(node)
                    prev.add_children(node)

            if c == 0:
                c = len(self._q)
                cur_depth += 1

    def predict(self, x):
        return self._tree.walk(x)


if __name__ == '__main__':
    # data = [
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 2, 1],
    #     [2, 1, 1, 1, 2],
    #     [3, 2, 1, 1, 2],
    #     [3, 3, 2, 1, 2],
    #     [3, 3, 2, 2, 1],
    #     [2, 3, 2, 2, 2],
    #     [1, 2, 1, 1, 1],
    #     [1, 3, 2, 1, 2],
    #     [3, 2, 2, 1, 2],
    #     [1, 2, 2, 2, 2],
    #     [2, 2, 1, 2, 2],
    #     [2, 1, 2, 1, 2],
    #     [3, 2, 1, 2, 1],
    # ]
    # data = np.array(data)
    # x = data[:, :-1]
    # y = data[:, -1]

    df = pd.read_csv('kr-vs-kp_2_filtered.csv')
    data = df.to_numpy()
    data = data[:, 1:]
    folds = 5
    X = data[:, :-1]
    y = data[:, -1]
    p_grid = {'max_depth': np.arange(10, 1000, 20), 'thresh': np.arange(1e-7, 0, 9e-7)}

    # mask = np.random.randint(y.shape[0], size=(int(y.shape[0] * 0.2), 1))
    # y[mask] = (y[mask] + 1) % 2
    # enumerate splits
    outer_results = list()
    # r = np.arange(1e-4, 0, - 1e-5)
    r = np.arange(1, 10)

    average_test_accs = np.zeros(r.shape[0])
    average_train_accs = np.zeros(r.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = ID3(max_depth=3, thresh=None)
    model.fit(X_train, y_train)

    yhat = model.predict(X_test)

    print(confusion_matrix(y_test, yhat))

    dsp = plot_confusion_matrix(model, X_test, y_test,
                          display_labels=['won', 'nowin'],
                          cmap=plt.cm.Blues, values_format = '.0f')

    dsp.ax_.set_title("Confusion matrix re-implementation")
    plt.show()
