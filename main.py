from pandas import read_csv, DataFrame
import numpy as np

from sklearn.model_selection import *
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def read_data_from_file(file_path: str, separator: str = ","):
    try:
        data = read_csv(file_path, sep=separator)
        return data
    except IOError:
        print('Reading data from file failed')


def prepare_data(df: DataFrame):
    df = df.drop(["Type/Index"], axis=1)
    df.drop_duplicates(inplace=True)

    df.replace("Explosive", 1, inplace=True)
    df.replace("Non-Explosive", 0, inplace=True)

    for key in df.keys()[1:]:
        df[key] = df[key].replace(",", ".", regex=True)
        df[key] = df[key].astype(float)

    x = df.drop(["Label"], axis=1).to_numpy()
    y = df["Label"].to_numpy()

    return train_test_split(x, y, test_size=0.25, random_state=1, stratify=y)


def decision_tree(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier(criterion='gini', random_state=1)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    print(f"Decision tree accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class predict: {(y_pred != y_test).sum()}\n")


def random_forest(x_train, x_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=25, criterion='gini', n_jobs=4, random_state=1)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    print(f"Random forest accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class predict: {(y_pred != y_test).sum()}\n")


def svm_classifier(x_train, x_test, y_train, y_test):
    svc = SVC(C=1.0, kernel='rbf', random_state=1)
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print(f"SVN accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class predict: {(y_pred != y_test).sum()}\n")


def knn_classifier(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(f"KNN accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class predict: {(y_pred != y_test).sum()}\n")


def main():
    data_path = "data/data.csv"
    df = read_data_from_file(data_path, separator=";")

    x_train, x_test, y_train, y_test = prepare_data(df)

    print(x_train.shape)
    print(x_test.shape)

    decision_tree(x_train, x_test, y_train, y_test)
    random_forest(x_train, x_test, y_train, y_test)
    svm_classifier(x_train, x_test, y_train, y_test)
    knn_classifier(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
