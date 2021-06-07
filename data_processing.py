from pandas import read_csv, DataFrame
from sklearn.model_selection import *


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