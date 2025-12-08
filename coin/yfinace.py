import datetime
import io

import yfinance as yf
import matplotlib.pyplot as plt


class Sol:
    def __init__(self):
        pass

    @staticmethod
    def finance(coin, start: str) -> bytes:
        end = Sol.localtime()
        data = yf.download(coin, start=start, end=end, progress=False, timeout=10)

        if data.empty:
            raise ValueError("無法獲取資料，請檢查日期")

        print(data.head())
        print(data.tail(),data["Close"])
        lastest_close = data["Close"].iloc[-1]
        plt.rcParams["font.family"] = "Microsoft YaHei"
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data["Close"], label="收盤價", color="orange")
        plt.title(f"{coin}歷史價格走勢圖 最新收盤價 {lastest_close}")
        plt.xlabel("日期")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(True)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()

        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def localtime():
        today = datetime.date.today()
        format_date = today.strftime("%Y-%m-%d")

        return format_date
