import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def coingecko(start, end):
    coin = "SOL-USD"

    data = yf.download(coin, start=start, end=end)

    print(data.head())

    plt.rcParams["font.family"] = "Microsoft YaHei"

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="收盤價", color="orange")
    plt.title(f"{coin}歷史價格走勢圖")
    plt.xlabel("日期")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(True)
    plt.show()

coingecko("2023-01-01","2025-11-15")