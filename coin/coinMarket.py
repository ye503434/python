import json
import requests as rq
import time
import datetime
import pandas as pd
from bs4 import BeautifulSoup


def date_to_timestamp(s, e):
    start = str(time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()))[:-2]
    end = str(time.mktime(datetime.datetime.strptime(e, "%d/%m/%Y").timetuple()) + 86400)[:-2]
    print(start + "     " + end)
    return start, end


def coinMarket(start_date, end_date):
    url = "https://coinmarketcap.com/"
    response = rq.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    data = soup.find('script', id="__NEXT_DATA__", type="application/json")
    coins = {}
    coin_data = json.loads(data.text)
    initial_state = coin_data.get("props", {}).get("initialState", {})

    if not isinstance(initial_state, dict) or 'cryptocurrency' not in initial_state:
        print(" 無法找到 'initialState' 或 'cryptocurrency'。")
        listings = []
    else:
        listings = initial_state["cryptocurrency"]['listingLatest']['data']
    # listings = coin_data["props"]["initialState"]["cryptocurrency"]['listingLatest']['data']

    for i in listings[1:2]:
        coins[str(i[8])] = i[15]

    start, end = date_to_timestamp(start_date, end_date)

    percent = 0
    total = 100

    for coin in coins:
        Market_Cap = []
        Open = []
        High = []
        Low = []
        Volume = []
        Close = []
        Date = []
        try:

            url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id=" + coin + "&convertId=2781&timeStart=" + start + "&timeEnd=" + end

            response = rq.get(url)
            #    soup = BeautifulSoup(response.text, "html.parser")
            history_data = json.loads(response.text)
            quotes = history_data["data"]['quotes']
            for quote in quotes:
                time.sleep(0.01)
                Market_Cap.append(quote["quote"]["marketCap"])
                Open.append(quote["quote"]["open"])
                Date.append(quote["quote"]["timestamp"][:10])
                High.append(quote["quote"]["high"])
                Low.append(quote["quote"]["low"])
                Volume.append(quote["quote"]["volume"])
                Close.append(quote["quote"]["close"])

            df = pd.DataFrame(
                columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'])  # All Coins' Data
            df['Date'] = Date
            df['Open'] = Open
            df['High'] = High
            df['Low'] = Low
            df['Close'] = Close
            df['Volume'] = Volume
            df['Market Cap'] = Market_Cap
            print(df)
        except Exception as e:
            print(f"數據發生錯誤:{e}")
        percent += 1
        print('\r' + '[Web Scraping]:[%s%s]%.2f%%;' % ('█' * int(percent * 20 / total),
                                                       ' ' * (20 - int(percent * 20 / total)),
                                                       float(percent / total * 100)), end='')


coinMarket("30/10/2024", "30/3/2025")
