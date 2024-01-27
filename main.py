# libraries
import argparse
import yfinance as yf
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error
import os
import datetime

# custom libraries
from scaling.scaler import CustomScaler
from splitting.splitter import CustomSplitter
from modelling.model import CustomModel

# types
from pandas import DataFrame
from argparse import Namespace
from numpy import ndarray
from typing import Optional

input_length = 10
target_length = 1

mae_errors = []
mape_errors = []

warnings.filterwarnings('ignore')

def download_data(
        symbol: str,
        period: str,
        interval: str
) -> DataFrame:
    return yf.download(symbol, period=period, interval=interval)

def preprocess_data(
        df: DataFrame,
        target_length: int,
        desired_columns:list=None
    ) -> ndarray:
    
    df['Target'] = df['Close'].shift(-target_length)
    df.drop('Adj Close', axis=1, inplace=True)
    df = df.dropna()

    # calculate adx
    df = calculate_adx_di(df)

    # if given
    if desired_columns:
        df = df[desired_columns].values
    return df


def calculate_adx_di(
        df: DataFrame,
        len_param:int=14
    ) -> DataFrame:
    """
    Verilen bir DataFrame'e ADX ve DIPlus, DIMinus sütunlarını ekleyen fonksiyon.

    Parameters:
    - df: DataFrame, Hissedarlık verilerini içeren DataFrame.
    - len_param: int, SmoothedTrueRange ve ADX hesaplamak için kullanılacak periyot.

    Returns:
    - df: DataFrame, Güncellenmiş DataFrame.
    """

    # TrueRange hesaplama
    df['TrueRange'] = df.apply(lambda row: max(row['High'] - row['Low'], abs(row['High'] - df['Close'].shift(1)[row.name]), abs(row['Low'] - df['Close'].shift(1)[row.name])), axis=1)

    # Directional Movement hesaplama
    df['DirectionalMovementPlus'] = df.apply(lambda row: max(row['High'] - df['High'].shift(1)[row.name], 0) if row['High'] - df['High'].shift(1)[row.name] > df['Low'].shift(1)[row.name] - row['Low'] else 0, axis=1)
    df['DirectionalMovementMinus'] = df.apply(lambda row: max(df['Low'].shift(1)[row.name] - row['Low'], 0) if df['Low'].shift(1)[row.name] - row['Low'] > row['High'] - df['High'].shift(1)[row.name] else 0, axis=1)

    # # Smoothed TrueRange ve Directional Movement hesaplama
    df['SmoothedTrueRange'] = 0.0
    df['SmoothedDirectionalMovementPlus'] = 0.0
    df['SmoothedDirectionalMovementMinus'] = 0.0

    for i in range(1, len(df)):
        df.at[df.index[i], 'SmoothedTrueRange'] = df.at[df.index[i-1], 'SmoothedTrueRange'] - (df.at[df.index[i-1], 'SmoothedTrueRange'] / len_param) + df.at[df.index[i], 'TrueRange']
        df.at[df.index[i], 'SmoothedDirectionalMovementPlus'] = df.at[df.index[i-1], 'SmoothedDirectionalMovementPlus'] - (df.at[df.index[i-1], 'SmoothedDirectionalMovementPlus'] / len_param) + df.at[df.index[i], 'DirectionalMovementPlus']
        df.at[df.index[i], 'SmoothedDirectionalMovementMinus'] = df.at[df.index[i-1], 'SmoothedDirectionalMovementMinus'] - (df.at[df.index[i-1], 'SmoothedDirectionalMovementMinus'] / len_param) + df.at[df.index[i], 'DirectionalMovementMinus']

    # DIPlus, DIMinus ve ADX hesaplama
    df['DIPlus'] = df['SmoothedDirectionalMovementPlus'] / df['SmoothedTrueRange'] * 100
    df['DIMinus'] = df['SmoothedDirectionalMovementMinus'] / df['SmoothedTrueRange'] * 100
    df['DX'] = abs(df['DIPlus'] - df['DIMinus']) / (df['DIPlus'] + df['DIMinus']) * 100
    df['ADX'] = df['DX'].rolling(window=len_param).mean()

    # Gereksiz sütunları düşürme
    df = df.drop(['TrueRange', 'DirectionalMovementPlus', 'DirectionalMovementMinus', 'SmoothedTrueRange', 'SmoothedDirectionalMovementPlus', 'SmoothedDirectionalMovementMinus', 'DX'], axis=1)

    # İlk 14 satırın NaN değerlerini temizleme
    df = df.dropna()

    return df


def draw_model_history(
        model_history:dict
    ) -> None:
    plt.plot(model_history.history['loss'], label='Training Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def draw_and_save_predictions(
        preds: ndarray,
        ground_truth: ndarray
    ) -> None:
    x_values = np.arange(len(preds))

    plt.figure(figsize=(20, 20))

    # Çizgi grafikleri oluştur
    plt.plot(x_values, preds, label='Predictions', marker='o')
    plt.plot(x_values, ground_truth, label='Actual', marker='o')

    # Eksen etiketleri
    plt.xlabel('Örnek Numarası')
    plt.ylabel('Değerler')

    # Başlık
    plt.title('Predictions vs Actuals')

    # İlgili yeri göster
    plt.legend()

    plt.savefig('example_plot.png')

    # Grafik göster
    plt.show()

def run_experiment(exp_id):
    print(f"Experiment {exp_id + 1}")
    main(args)

    # main fonksiyonundan global değişkenlere erişelim
    mae_error = mae_error_global
    mape_error = mape_error_global

    # Her bir deneyin sonuçlarını logla
    with open(file_name, "a") as log_file:
        log_file.write(f"Experiment {exp_id + 1}:\n")
        log_file.write(f"Mean Absolute Error: {mae_error}\n")
        log_file.write(f"Mean Absolute Percentage Error: {mape_error}\n")
        log_file.write("-" * 50 + "\n")


def main(args: Namespace):

    # AAPL hisse senedi sembolü
    symbol = "AAPL"
    period = "5y"
    interval = "1d"

    data = download_data(symbol, period, interval)

    # TODO: not implemented in ipynb, discussable 
    # desired_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'DIPlus', 'DIMinus', 'ADX']

    # preprocess and calculate adx with selecting desired columns
    data  = preprocess_data(data, target_length)

    scaler = CustomScaler(
        MinMaxScaler(feature_range=(0, 1))
    )

    scaled_data = scaler.scale(data)
    
    train_size = int(len(scaled_data) * 0.8)
    splitter = CustomSplitter(
        train_size=train_size,
        input_length=input_length
    )
    X_train, y_train, X_test, y_test = splitter.split(scaled_data)


    _temp = "Splitted Shapes"
    # print(f"\n{'*'*25}{_temp}{'*'*25}")
    # print(f"X_train.shape -> {X_train.shape}")
    # print(f"y_train.shape -> {y_train.shape}")
    # print(f"X_test.shape -> {X_test.shape}")
    # print(f"y_test.shape -> {y_test.shape}")
    # print(f"{'*'*(50+len(_temp))}\n")

    # reshape for keras
    X_train = X_train.reshape((X_train.shape[0], input_length, X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], input_length, X_test.shape[2]))

    # Eğitim ve test setlerini göster
    _temp = "After Reshape"
    # print(f"\n{'*'*25}{_temp}{'*'*25}")
    # print("Train set:", X_train.shape, y_train.shape)
    # print("Test set:", X_test.shape, y_test.shape)
    # print(f"{'*'*(50+len(_temp))}\n")
    
    modeller = CustomModel()
    layers = [
        LSTM(100, input_shape=(
            X_train.shape[1], 
            X_train.shape[2]), 
            return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ]
    
    params = {
        "loss": "mae",
        "optimizer": "adam"
    }
    # build and compile model
    modeller.build_sequential_model(layers, params)
    
    
    _temp = "Model Summary"
    # print(f"\n{'*'*25}{_temp}{'*'*25}")
    # print(modeller.get_model().summary())
    # print(f"{'*'*(50+len(_temp))}\n")

    checkpoint = ModelCheckpoint(
        'best_model.h5', 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )
    early_stopping = EarlyStopping(patience=5)
    modeller.set_callbacks([
        checkpoint,
        early_stopping
    ])

    fit_params = {
        "epochs": 30,
        "batch_size": 1,
        "verbose": 1
    }
    modeller.fit(
        train_data=[X_train, y_train],
        test_data= [X_test, y_test],
        params= fit_params
    )

    # draw_model_history(modeller.get_history())

    preds = modeller.predict(X_test)

    # draw_and_save_predictions(preds, ground_truth=y_test)

    # TODO: Dinamik olmalı
    # calculate error 
    # calculate MAE 
    mae_error = mae(preds, y_test)
    print("Mean absolute error : " + str(mae_error)) 

    mape_error = mean_absolute_percentage_error(preds, y_test)
    print("Mean absolute Percentage error : " + str(mape_error)) 

    # Bu değerleri fonksiyon dışında kullanabilmek için global değişkenlere atayalım.
    global mae_error_global, mape_error_global
    mae_error_global = mae_error
    mape_error_global = mape_error

    mae_errors.append(mae_error)
    mape_errors.append(mape_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", type=str, default="config.yaml")

    # input_length = 10
    # target_length = 1

    args = parser.parse_args()

    num_experiments = 10

    # Başka bir dizinde bir klasör oluştur
    output_directory = "experiment_results"
    os.makedirs(output_directory, exist_ok=True)

    # Her çalıştırmada farklı bir dosya adı oluştur
    current_run = 1
    file_name = os.path.join(output_directory, f"experiment_results_run_{current_run}.txt")
    while os.path.exists(file_name):
        current_run += 1
        file_name = os.path.join(output_directory, f"experiment_results_run_{current_run}.txt")

    for exp_id in range(num_experiments):
        run_experiment(exp_id)

    # Tüm deneylerin ortalamasını hesapla
    avg_mae = sum(mae_errors) / num_experiments
    avg_mape = sum(mape_errors) / num_experiments

   # Ortalamaları aynı dosyaya yazdır
    with open(file_name, "a") as log_file:
        log_file.write(f"Average Mean Absolute Error: {avg_mae}\n")
        log_file.write(f"Average Mean Absolute Percentage Error: {avg_mape}\n")
        log_file.write("=" * 50 + "\n")