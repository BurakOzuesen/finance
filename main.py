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
import pandas as pd

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
target_length = 10

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
        model_history: dict,
        file_path: str
    ) -> None:
    plt.plot(model_history.history['loss'], label='Training Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path)
    plt.close()

def draw_and_save_predictions(
        preds: ndarray,
        ground_truth: ndarray,
        file_path: str
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

    plt.savefig(file_path)
    plt.close()


def run_experiment(exp_id, exp_dir):
    print(f"Experiment {exp_id + 1}")
    main(args, exp_dir)

    # main fonksiyonundan global değişkenlere erişelim
    mae_error = mae_error_global
    mape_error = mape_error_global

    # Her bir deneyin sonuçlarını logla
    log_file_path = os.path.join(exp_dir, "experiment_results.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Experiment {exp_id + 1}:\n")
        log_file.write(f"Mean Absolute Error: {mae_error}\n")
        log_file.write(f"Mean Absolute Percentage Error: {mape_error}\n")
        log_file.write("-" * 50 + "\n")


def main(args: Namespace, exp_dir: str):

    # AAPL hisse senedi sembolü
    # symbol = "AAPL"
    # period = "1y"
    # interval = "1d"

    # data = download_data(symbol, period, interval)

    data = pd.read_csv("./stock_data/5m/original/ISCTR.IS_5m_original.csv", index_col=0)
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
        "epochs": 3,
        "batch_size": 32,
        "verbose": 1
    }
    modeller.fit(
        train_data=[X_train, y_train],
        test_data= [X_test, y_test],
        params= fit_params
    )

    # Model geçmişini çiz ve kaydet
    history_file_path = os.path.join(exp_dir, "history.png")
    draw_model_history(modeller.get_history(), history_file_path)

    preds = modeller.predict(X_test)


    # Tahminleri çiz ve kaydet
    predictions_file_path = os.path.join(exp_dir, "predictions.png")
    draw_and_save_predictions(preds, ground_truth=y_test, file_path=predictions_file_path)

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

    num_experiments = 2

    # Ana klasörü oluştur
    output_directory = "experiment_results"
    os.makedirs(output_directory, exist_ok=True)

    # Her çalıştırmada farklı bir dosya adı oluştur
    current_run = 1
    base_dir = os.path.join(output_directory, f"run_{current_run}")
    while os.path.exists(base_dir):
        current_run += 1
        base_dir = os.path.join(output_directory, f"run_{current_run}")
    os.makedirs(base_dir)

    all_mae_errors = []
    all_mape_errors = []

    for exp_id in range(num_experiments):
        exp_dir = os.path.join(base_dir, f"experiment_{exp_id+1}")
        os.makedirs(exp_dir)
        mae_errors = []
        mape_errors = []
        run_experiment(exp_id, exp_dir)
        all_mae_errors.append(mae_errors)
        all_mape_errors.append(mape_errors)

    # Tüm deneylerin ortalamasını hesapla
    avg_mae = sum([sum(errors) for errors in all_mae_errors]) / (num_experiments * len(all_mae_errors[0]))
    avg_mape = sum([sum(errors) for errors in all_mape_errors]) / (num_experiments * len(all_mape_errors[0]))

    # Ortalamaları aynı dosyaya yazdır
    log_file_path = os.path.join(base_dir, "average_results.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Average Mean Absolute Error: {avg_mae}\n")
        log_file.write(f"Average Mean Absolute Percentage Error: {avg_mape}\n")
        log_file.write("=" * 50 + "\n")
        log_file.write("Experiment Results:\n")
        for i, (mae_errors, mape_errors) in enumerate(zip(all_mae_errors, all_mape_errors), start=1):
            log_file.write(f"Experiment {i}:\n")
            log_file.write(f"Mean Absolute Error: {sum(mae_errors) / len(mae_errors)}\n")
            log_file.write(f"Mean Absolute Percentage Error: {sum(mape_errors) / len(mape_errors)}\n")
            log_file.write("-" * 50 + "\n")