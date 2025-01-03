import yfinance as yf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Logging yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# S&P 500 ve İş Bankası hisse senetlerinin simgeleri
tickers = ['^GSPC', 'ISCTR.IS']

# Farklı zaman dilimleri ve bu zaman dilimleri için uygun dönemler
intervals_periods = {
    '1m': '7d',     # 1 dakika: Son 7 gün
    '5m': '60d',    # 5 dakika: Son 60 gün
    '1h': '730d',   # 1 saat: Son 730 gün (2 yıl)
    '1d': '5y',     # 1 gün: Son 5 yıl
    '1wk': '10y'    # 1 hafta: Son 10 yıl
}

# Ana klasör
main_folder = 'stock_data'

def create_folder(path):
    """Klasör oluşturma fonksiyonu"""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Folder created: {path}")

def save_data(data, folder, file_name):
    """Veriyi CSV olarak kaydetme fonksiyonu"""
    file_path = os.path.join(folder, file_name)
    data.to_csv(file_path)
    logging.info(f"Data saved to {file_path}")

def split_and_save_data(data, interval_folder, ticker, interval):
    """Veriyi %60 eğitim, %20 doğrulama ve %20 test setlerine ayırıp kaydetme fonksiyonu"""
    train_data, temp_data = train_test_split(data, test_size=0.4, shuffle=False)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)
    
    # Alt klasörler oluştur
    train_folder = os.path.join(interval_folder, 'train')
    validation_folder = os.path.join(interval_folder, 'validation')
    test_folder = os.path.join(interval_folder, 'test')
    
    create_folder(train_folder)
    create_folder(validation_folder)
    create_folder(test_folder)
    
    # Dosya isimlerini oluştur
    save_data(train_data, train_folder, f"{ticker.replace('^', '')}_{interval}_train.csv")
    save_data(validation_data, validation_folder, f"{ticker.replace('^', '')}_{interval}_validation.csv")
    save_data(test_data, test_folder, f"{ticker.replace('^', '')}_{interval}_test.csv")

def download_and_process_data(ticker, interval, period):
    """Veriyi indir ve işle"""
    try:
        logging.info(f"Downloading {ticker} data for interval {interval} and period {period}")
        data = yf.download(ticker, interval=interval, period=period)
        
        # Zaman dilimi klasör ismi oluştur
        interval_folder = os.path.join(main_folder, interval)
        create_folder(interval_folder)
        
        # Orijinal veriyi kaydet
        original_folder = os.path.join(interval_folder, 'original')
        create_folder(original_folder)
        save_data(data, original_folder, f"{ticker.replace('^', '')}_{interval}_original.csv")
        
        # Veriyi böl ve kaydet
        split_and_save_data(data, interval_folder, ticker, interval)
    
    except Exception as e:
        logging.error(f"Error downloading data for {ticker} with interval {interval}: {e}")

def main():
    create_folder(main_folder)
    
    for ticker in tickers:
        for interval, period in intervals_periods.items():
            download_and_process_data(ticker, interval, period)

if __name__ == "__main__":
    main()
