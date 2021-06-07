from collections import deque

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
        res = []
        for i in range(x.shape[0]):
            node = self.root
            res.append(self.walk_util(node, x[i, :]))
        return res

    def walk_util(self, node, s):
        # Leaf node
        if len(node.children) == 0:
            return node.data[2]

        for child in node.children:
            if child.data[1] == s[node.data[0]]:
                return self.walk_util(child, s)


class ID3:
    def __init__(self, thresh=None, max_depth=None):
        # self.c = y.unique()
        self._tree = DecisionTree()
        self._q = deque()
        self._tree_q = deque()
        if thresh is not None:
            self._thresh = 1.0 - thresh
        else:
            self._thresh = None
        self._max_depth = max_depth

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
        if self._thresh is not None and h_min > self._thresh:
            return 0, -1, np.array([])
        return h_min, h_min_index, min_child

    def fit(self, data):
        cur_depth = 0
        c = 1
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
            if self._max_depth is None or (
                    self._max_depth is not None and cur_depth <= self._max_depth):
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

    df = pd.read_csv('mushrooms_filtered.csv')
    data = df.to_numpy()
    data = data[:, 1:]
    train, test = train_test_split(data, test_size=0.2)

    id3 = ID3(max_depth=2)
    id3.fit(train)

    X_test, y_test = test[:, :-1], test[:, -1]
    acc = id3.predict(X_test) == y_test
    acc = acc.mean()
    print(acc)
# 0.9261538461538461
