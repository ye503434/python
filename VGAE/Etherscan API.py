import os

import pandas as pd
import requests
import time

from dotenv import load_dotenv
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
# 把Top0.1%前5000筆資料 拿去以太坊尋找是否是合約地址，最後產出5000筆分類好的地址csv。
ApiKey = os.getenv('ETHERSCAN_API_KEY')
topDf = pd.read_csv("../eth_anomalies_report_v5.csv")

#Retry機制，因為會跑出Read time out 這個錯誤log。
session = requests.Session()
retry_strategy = Retry(
    total = 3,
    backoff_factor = 1,
    status_forcelist=[429,500,502,503,504]#錯誤狀態代碼
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount('https://', adapter)
session.mount('http://', adapter)

target = 5000
subsetDf = topDf['Wallet_Address'][:target]
result = []
outputFile = '../top_verification_results.csv'

for addr in tqdm(subsetDf, desc='requestETHApi'):
    try:
        url = f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={addr}&apikey={ApiKey}"

        response = session.get(url, timeout=20).json()

        label = "Unknown"
        is_contract = "No"

        if response.get('status') == '1' and response['result'][0].get('ContractName'):
            label = response['result'][0]['ContractName']
            is_contract = 'Yes'

        result.append({
            'address': addr,
            'etherscan_label': label,
            'is_contract': is_contract
        })
    except Exception as e:
        print(f'\n{addr}發生錯誤{e}')
        result.append({
            'address': addr,
            'etherscan_label': "Error",
            'is_contract': "Error"
        })
        time.sleep(1)
        continue
    time.sleep(0.3)
pd.DataFrame(result).to_csv('../top_verification_results.csv', index=False)
print("完成")
