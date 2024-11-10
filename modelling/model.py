
# libraries
from keras import Sequential
from keras.models import Model

# custom libraries
from utils import apply_functions_consecutively

# types
from pandas import DataFrame
from typing import List
from numpy import ndarray
from keras.layers import Dense, concatenate, Input


# TODO: Dynamic type handling must be added!
class CustomModel:
    model = None
    callbacks = None
    model_history = None
    preds = None

    def __init__(self) -> None:
        self.callbacks=list()

    def build_sequential_model(
        self,
        layers: list,
        params: dict
    ) -> None:
        self.model = Sequential(layers)
        self.model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=params["metrics"])

    def build_custom_model(
            self, 
            inputs: Input,
            output_layers: List[Dense],
            layers: list,
            params: dict
            
    ) -> None:
        out_cont, out_prob  = output_layers
        # concatenate([output_continuous, output_probability], name='concatenated_outputs')
        
        shared_output = apply_functions_consecutively(inputs, layers)
        
        # continous
        output_continuous = out_cont(shared_output)
        # binary
        output_probability = out_prob(shared_output)

        # outputs = concatenate()

        self.model = Model(inputs=inputs, outputs=[output_continuous, output_probability])
        self.model.compile(
            loss=params['loss'], 
            optimizer=params['optimizer'], 
            metrics=params["metrics"])
        


    def set_callbacks(
            self, 
            callbacks:list
        ) -> None:
        self.callbacks.extend(callbacks)

    def get_model(self) -> Sequential:
        return self.model
    def get_history(self) -> dict:
        return self.model_history

    def fit(
            self,
            train_data: List[DataFrame],
            test_data: List[DataFrame],
            params: dict
        ) -> None:

        """
        params: {
            epochs: 30,
            batch_size: 1,
            verbose: 1
        }
        """

        self.model_history = self.model.fit(
            train_data[0], # X_train 
            train_data[1], # y_train
            epochs=params["epochs"], 
            batch_size=params["batch_size"], 
            validation_data=(
                test_data[0], # X_test
                test_data[1]  # y_test
            ), 
            verbose=params["verbose"], 
            callbacks=self.callbacks
        )

    def predict(
            self, 
            data: ndarray
        ) -> ndarray:
        self.preds = self.model.predict(data)
        return self.preds
