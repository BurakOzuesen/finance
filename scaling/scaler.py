# libraries
import pandas as pd

# types
from pandas import DataFrame

# TODO: Dynamic type handling must be added!
class CustomScaler:
    scaler=None

    def __init__(self, scaler) -> None:
        self.scaler = scaler

    def scale(
        self, 
        data: DataFrame
    ) -> DataFrame:
        
        return pd.DataFrame(
            self.scaler.fit_transform(data), 
            columns=data.columns, 
            index=data.index
        )