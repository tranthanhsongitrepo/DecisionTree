from id3 import Id3Estimator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
if __name__ == "__main__":
    # df = pd.read_csv('kr-vs-kp_2.csv')
    #
    # # df = df.apply(lambda x: np.nan if x == '?' else x)
    # df = df.fillna(df.mode().iloc[0])
    #
    # for col in df:
    #     le = LabelEncoder()
    #     df[col] = le.fit_transform(df[col])
    #
    # df.to_csv('kr-vs-kp_2_filtered.csv')

    df = pd.read_csv('kr-vs-kp_2_filtered.csv')
    data = df.to_numpy()
    data = data[:, 1:]
    folds = 5
    X = data[:, :-1]
    y = data[:, -1]

    outer_results = list()
    # r = np.arange(1e-4, 0, - 1e-5)
    r = np.arange(1, 10)
    average_test_accs = np.zeros(r.shape[0])
    average_train_accs = np.zeros(r.shape[0])

    for i, thresh in enumerate(r):
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        average_test_acc = 0.0
        average_train_acc = 0.0
        for train_ix, test_ix in cv_outer.split(X):
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            model = Id3Estimator(max_depth=thresh)
            model.fit(X_train, y_train)
            yhat = model.predict(X_test)
            # evaluate the model
            acc = accuracy_score(y_test, yhat)

            average_test_acc += acc

            yhat = model.predict(X_train)
            # evaluate the model
            acc = accuracy_score(y_train, yhat)
            # store the result
            average_train_acc += acc

        average_test_accs[i] = average_test_acc / 10
        average_train_accs[i] = average_train_acc / 10

    plt.plot(r, average_train_accs, label='train')
    plt.plot(r, average_test_accs, label='test')
    plt.xlabel("IG")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
