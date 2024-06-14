import yfinance as yf
import os

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

# Klasör oluşturma fonksiyonu
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Ana klasörü oluştur
create_folder(main_folder)

# Her hisse senedi ve her zaman dilimi için verileri indir ve kaydet
for ticker in tickers:
    for interval, period in intervals_periods.items():
        # Veriyi indir
        print("*"*50)
        print(f"Downloading {ticker} data for interval {interval} and period {period}")

        data = yf.download(ticker, interval=interval, period=period)
        print("*"*50)
        
        # Klasör ismi oluştur
        interval_folder = os.path.join(main_folder, interval)
        create_folder(interval_folder)
        
        # Dosya ismi oluştur
        file_name = f"{ticker.replace('^', '')}_{interval}.csv"
        file_path = os.path.join(interval_folder, file_name)
        
        # Veriyi CSV olarak kaydet
        data.to_csv(file_path)
        print(f"Data saved to {file_path}")
