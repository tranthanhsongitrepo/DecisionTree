from collections import deque

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from matplotlib import pyplot as plt

class Node:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_children(self, children):
        self.children.append(children)


class DecisionTree:
    def __init__(self):
        self.root = Node([None, -1, -1])

    def walk(self, x):
        res = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            node = self.root
            res[i] = int(self.walk_util(node, x[i, :]))
        return res

    def walk_util(self, node, s):
        # Leaf node
        if len(node.children) == 0:
            return node.data[2]

        for child in node.children:
            if child.data[1] == s[node.data[0]]:
                return self.walk_util(child, s)


class ID3(BaseEstimator):
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
    p_grid = {'max_depth': np.arange(10, 1000, 20), 'thresh': np.arange(1e-7, 1e-6, 9e-7)}

    mask = np.random.randint(y.shape[0], size=(int(y.shape[0] * 0.2), 1))
    y[mask] = (y[mask] + 1) % 2
    # enumerate splits
    outer_results = list()
    average_test_accs = np.zeros(50)
    average_train_accs = np.zeros(50)

    for i in range(50):
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        average_test_acc = 0.0
        average_train_acc = 0.0
        for train_ix, test_ix in cv_outer.split(X):
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            # configure the cross-validation procedure
            # cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
            # define the model
            model = ID3(max_depth=i, thresh=None)
            model.fit(X_train, y_train)
            #
            # # define search
            # search = GridSearchCV(model, p_grid, scoring='accuracy', cv=cv_inner, refit=True)
            # # execute search
            # result = search.fit(X_train, y_train)
            # # get the best performing model fit on the whole training set
            # best_model = result.best_estimator_
            # # evaluate model on the hold out dataset
            yhat = model.predict(X_test)
            # evaluate the model
            acc = accuracy_score(y_test, yhat)

            average_test_acc += acc

            yhat = model.predict(X_train)
            # evaluate the model
            acc = accuracy_score(y_train, yhat)
            # store the result
            average_train_acc += acc

            # report progress
            # print(result.best_params_, result.best_score_)

        average_test_accs[i] = average_test_acc / 10
        average_train_accs[i] = average_train_acc / 10


# 0.9261538461538461

    plt.plot(np.arange(50), average_test_accs, label='train')
    plt.plot(np.arange(50), average_test_accs, label='test')
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.show()
