import numpy as np
import pandas as pd
import os


def load_ML_CUP21():

    """
    Loads ML_CUP21 dataset for regression tasks benchmarking.
    
    Each line of the dataset file is in the following form:
    "idx i_0 i_1 i_2 i_3 i_4 i_5 i_6 i_7 i_8 i_9 target_x target_y",
    where idx is the index, i_i is the i-th input, and target_x and target_y are the continuous values to predict.

    """

    dataset = pd.read_csv(f"{os.path.abspath(os.path.dirname(__file__))}/datasets/ML-CUP21.csv", usecols = range(1,13), header=None)
    colums = [f"i_{i}" for i in range(1,11)] + ["target_x", "target_y"]
    dataset.columns = colums
    X = dataset.drop(columns = ["target_x","target_y"]).values
    y = dataset[["target_x", "target_y"]].values
    return X, y


def load_monks(idx, partition):

    """
    Loads the idx-th monks dataset for binary classification tasks benchmarking.
    partition parameter must be either "train" or "test", according to the dataset partition to be loaded.
    Each row of a monks dataset file is in the following form:
    "   l f_0 f_1 f_2 f_3 f_4 f_5 idx",
    where l is the label (0 or 1), f_i is the i-th feature, and idx is the index (each row starts with three blank spaces).
    The features of the dataset are categorical. In particular:
        f_0, f_1 and f_3 have three possible values (1, 2 or 3);
        f_2 and f_5 have two possible values (1 or 2);
        f_4 has four possible values (1, 2, 3 or 4).
    Each feature is one-hot encoded.

    """

    def one_hot_features_encoding(row):
        encoded_row = []
        for idx, value in enumerate(row):
            features_n_possible_values = {0: 3, 1: 3, 2: 2, 3: 3, 4: 4, 5: 2}
            encoded_feature = np.zeros(features_n_possible_values[idx]).tolist()
            encoded_feature[value - 1] = 1
            encoded_row += encoded_feature
        return encoded_row

    filepath = f"{os.path.abspath(os.path.dirname(__file__))}/datasets/monks-{idx}.{partition}"
    with open(filepath) as d:
        X = []
        y = []
        for line in d.readlines():
            line = line.lstrip()
            row = [int(x) for x in line.split(" ")[1:-1]]
            X.append(one_hot_features_encoding(row))
            label = [int(line[0])]
            y.append(label)
        X = np.array(X, dtype="float16")
        y = np.array(y, dtype="float16")

    return X, y
