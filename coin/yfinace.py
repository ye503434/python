import datetime

import yfinance as yf
import matplotlib.pyplot as plt


class Sol:
    def __init__(self):
        end = self.localtime()
        self.yfinance("2024-11-16", end)

    @staticmethod
    def yfinance(start: str, end: str):
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

    @staticmethod
    def localtime():
        today = datetime.date.today()
        format_date = today.strftime("%Y-%m-%d")

        return format_date


Sol()
