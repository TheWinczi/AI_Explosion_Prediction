from pandas import read_csv
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import *
from sklearn.metrics import accuracy_score


def read_data_from_file(file_path: str, separator: str = ","):
    try:
        data = read_csv(file_path, sep=separator)
        return data
    except IOError:
        print('Reading data from file failed')


def main():
    data_path = "data/explosive_data.csv"
    df = read_data_from_file(data_path, separator=";")
    df = df.drop(["Type/Index"], axis=1)
    df.drop_duplicates(inplace=True)

    df.replace("Explosive", 1, inplace=True)
    df.replace("Non-Explosive", 0, inplace=True)

    for key in df.keys()[1:]:
        df[key] = df[key].replace(",", ".", regex=True)
        df[key] = df[key].astype(float)

    x = df.drop(["Label"], axis=1).to_numpy()
    y = df["Label"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    print(y_test)
    print(np.unique(y_test))

    pnn = Perceptron(eta0=0.1, random_state=1)
    pnn.fit(x_train, y_train)
    y_pred = pnn.predict(x_test)
    print(accuracy_score(y_test, y_pred))




if __name__ == '__main__':
    main()