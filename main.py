from collections import deque

import numpy as np


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
    def __init__(self, x, thresh=1e-4):
        self.x = x
        # self.c = y.unique()
        self.tree = DecisionTree()
        self.q = deque([x])
        self.tree_q = deque([self.tree.root])
        if thresh is not None:
            self.thresh = thresh

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
        if h_min < self.thresh:
            return 0, -1, np.array([])
        return h_min, h_min_index, min_child

    def fit(self):
        while len(self.q) > 0:
            top = self.q.popleft()
            h_min, h_min_index, min_child = self.h(top)
            prev = self.tree_q.popleft()
            prev.data[0] = h_min_index
            y_uni, y_uni_count = np.unique(top[:, -1], return_counts=True)
            prev.data[2] = y_uni[y_uni_count.argmax()]
            for child in min_child:
                node = Node([None, child, -1])
                mask = top[:, h_min_index] == child
                self.q.append(top[mask])
                self.tree_q.append(node)
                prev.add_children(node)

    def predict(self, x):
        return self.tree.walk(x)


if __name__ == '__main__':
    data = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 2, 1],
        [2, 1, 1, 1, 2],
        [3, 2, 1, 1, 2],
        [3, 3, 2, 1, 2],
        [3, 3, 2, 2, 1],
        [2, 3, 2, 2, 2],
        [1, 2, 1, 1, 1],
        [1, 3, 2, 1, 2],
        [3, 2, 2, 1, 2],
        [1, 2, 2, 2, 2],
        [2, 2, 1, 2, 2],
        [2, 1, 2, 1, 2],
        [3, 2, 1, 2, 1],
    ]
    data = np.array(data)
    x = data[:, :-1]
    y = data[:, -1]

    id3 = ID3(data)
    id3.fit()
    print(id3.predict(x))
