from id3 import Id3Estimator
from id3 import export_graphviz
import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv('mushrooms.csv')

    # df = df.apply(lambda x: np.nan if x == '?' else x)
    df = df.fillna(df.mode().iloc[0])
    from sklearn.preprocessing import LabelEncoder

    for col in df:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df.to_csv('mushrooms_filtered.csv')

    train = df.sample(frac=0.8, random_state=200)  # random state is a seed value
    test = df.drop(train.index)

    X_train, X_test, y_train, y_test = train.iloc[:, :-1].reset_index(), test.iloc[:, :-1].reset_index(), train.iloc[:, -1], test.iloc[:, -1]
    tree = Id3Estimator()
    tree.fit(X_train, y_train)
    acc = (tree.predict(X_test) == y_test)
    acc = acc.mean()
    print(acc)
