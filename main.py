from classifiers import *
from data_processing import *


def main():
    data_path = "data/data.csv"
    df = read_data_from_file(data_path, separator=";")

    x_train, x_test, y_train, y_test = prepare_data(df)

    decision_tree(x_train, x_test, y_train, y_test)
    random_forest(x_train, x_test, y_train, y_test)
    svm_classifier(x_train, x_test, y_train, y_test)
    knn_classifier(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
