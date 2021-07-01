from id3 import Id3Estimator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import KFold, train_test_split
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Id3Estimator(max_depth=3)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    dsp = plot_confusion_matrix(model, X_test, y_test,
                                display_labels=['won', 'nowin'],
                                cmap=plt.cm.Blues, values_format='.0f')

    dsp.ax_.set_title("Confusion matrix lib")
    plt.show()
