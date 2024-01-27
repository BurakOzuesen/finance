# libraries
import numpy as np

# types
from pandas import DataFrame
from typing import Union
from numpy import ndarray

# TODO: Dynamic type handling must be added!
class CustomSplitter:
    train_size=None
    input_length=None

    def __init__(
            self, 
            train_size:int, 
            input_length: int
        ) -> None:

        self.train_size = train_size
        self.input_length = input_length

    def split(
            self, 
            data:DataFrame
        ) -> Union[ndarray, ndarray, ndarray, ndarray]: # X_train, y_train, X_test, y_test
        train, test = data.iloc[:self.train_size, :], data.iloc[self.train_size:, :]

        X_train, y_train = self.create_dataset(train, self.input_length)
        X_test, y_test = self.create_dataset(test, self.input_length)
        return X_train, y_train, X_test, y_test

    def create_dataset(
            self, 
            df: DataFrame, 
            time_steps:int=1
        ) -> Union[ndarray, ndarray]:
        # TODO: 'Target_Value' ve 'Target_Direction' column ismi dinamik al
    
        X, y = [], []
        for i in range(len(df) - time_steps):
            a = df.drop(['Target_Value'], axis=1).iloc[i:(i+time_steps), :].values
            X.append(a)
            y.append([df.iloc[i,5], df.iloc[i,6]])
        return np.array(X), np.array(y).reshape(-1, 2)


