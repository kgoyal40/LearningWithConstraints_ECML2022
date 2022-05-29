import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def data_processing(data):
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    categorical_features = list(data.select_dtypes(include='object').columns)
    categorical_features = list(set(categorical_features))
    numerical_features = [c for c in data.columns if c not in categorical_features]
    print(categorical_features)
    print(numerical_features)
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    for _c in categorical_features:
        data[_c] = pd.Categorical(data[_c])
    df_transformed = pd.get_dummies(data, drop_first=True)
    return df_transformed, scaler


def data_manipulation(X, y):
    y_violated = y.copy()
    indices_to_manipulate = [i for i in range(len(y)) if X['Credit_History'][i] == 0]
    indices_to_manipulate = random.sample(indices_to_manipulate, 35)
    for i in indices_to_manipulate:
        y_violated[i] = 1
    return X, y_violated
