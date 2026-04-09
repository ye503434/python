import pandas as pd
import requests
import time
from tqdm import tqdm

# 把Top0.1%前5000筆資料 拿去以太坊尋找是否是合約地址，最後產出5000筆分類好的地址csv。
ApiKey = "H1G34YSUJ58EDKQ2YJNNRXQUJCN9SN1TT6"
topDf = pd.read_csv("../eth_anomalies_report_v5.csv")

target = 5000
subsetDf = topDf['Wallet_Address'][:target]
result = []
outputFile = '../top_verification_results.csv'

for i, addr in enumerate(tqdm(subsetDf, desc='requestETHApi')):
    try:
        url = f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={addr}&apikey={ApiKey}"

        response = requests.get(url, timeout=10).json()

        label = "Unknown"
        is_contract = "No"

        if response.get('status') == '1' and response['result'][0].get('contractName'):
            label = response['result'][0]['ContractName']
            is_contract = 'Yes'

        result.append({
            'address': addr,
            'etherscan_label': label,
            'is_contract': is_contract
        })
    except Exception as e:
        print(f'{addr}發生錯誤{e}')
        continue
    time.sleep(0.2)
pd.DataFrame(result).to_csv('../top_verification_results.csv', index=False)
print("完成")
