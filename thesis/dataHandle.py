import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

file12to13 = r'D:\12000000to12999999_BlockTransaction.csv'
file11to12 = r'D:\11000000to11999999_BlockTransaction.csv'
chunkSize = 500000

# 把11-13的from到to的地址收集並且寫入二進位檔案。 結果: 共有50196418個唯一節點，邊數:382171280
uniqueAddr = set()

for filePath in [file11to12,file12to13]:
    print('掃描中......')
    total = 361 if "11000000" in filePath else 405
    for chunk in tqdm(pd.read_csv(filePath, chunksize=chunkSize, usecols=['from', 'to']), total= total):
        chunk = chunk.replace(r'^\s*$',np.nan,regex=True).dropna(subset = ['from','to'])
        uniqueAddr.update(chunk['from'].str.lower().str.strip().unique())
        uniqueAddr.update(chunk['to'].str.lower().str.strip().unique())

addrToId = {addr: i for i , addr in enumerate(uniqueAddr)}

with open('../addrMapCombine.pkl', 'wb') as f :
    pickle.dump(addrToId, f)
numNodes = len(addrToId)
print(f'掃描完成，總共{numNodes}唯一地址')

with open('../addrMapCombine.pkl', 'rb') as f:
    addrToId = pickle.load(f)
numNodes = len(addrToId)
print('初始化特徵矩陣和邊')
nodeFeatures = np.zeros((numNodes, 3), dtype=np.float32)
edgeList = []

for filePath in [file11to12, file12to13]:
    print(f'將數據轉換為ID格式:{filePath}')
    total = 361 if "11000000" in filePath else 405

    for chunk in tqdm(pd.read_csv(filePath, chunksize=chunkSize, usecols=['from', 'to', 'value']), total=total):
        # regex開啟正則表達，偵測到空改成nan，dropna可以刪除nan資料
        chunk = chunk.replace(r'^\s*$',np.nan,regex=True).dropna(subset = ['from','to'])
        #轉小寫、去除字串前後的空白字元
        srcLowerStrip = chunk['from'].str.lower().str.strip()
        dstLowerStrip = chunk['to'].str.lower().str.strip()

        # 轉換地址為ID 例如: 0x776a4012ba:0
        src = np.array([addrToId.get(a, -1) for a in srcLowerStrip], dtype=np.int64)
        dst = np.array([addrToId.get(a, -1) for a in dstLowerStrip], dtype=np.int64)
        #除錯，找不到回傳-1
        if (src == -1).any() or (dst == -1).any():
            # 跳出第一個找不到的地址
            if (src == -1).any():
                missing_val = srcLowerStrip[src == -1].iloc[0]
                col_name = "from"
            else:
                missing_val = dstLowerStrip[dst == -1].iloc[0]
                col_name = "to"

            print(f'\n偵錯，在 {col_name} 欄位發現缺漏地址: "{missing_val}"')
            print(f'字串長度: {len(str(missing_val))}')
            print(f'是否為空值: {pd.isna(missing_val)}')
            exit()

        #數值處理 Wei 轉 Ether 1e18 = 10的18次方
        #.values可以改成 .to_numpy() 更符合現代
        val = chunk['value'].values.astype(np.float32) / 1e18
        for s, d, v in zip(src, dst, val):
            nodeFeatures[s, 0] += 1  # 出度
            nodeFeatures[d, 1] += 1  # 入度
            nodeFeatures[s, 2] += v  # 總交易金額

        edgeList.append(np.stack([src, dst], axis=0))

print("\n正在合併邊")
edgeIndex = np.concatenate(edgeList, axis=1)
edgeIndexTorch = torch.from_numpy(edgeIndex).to(torch.long)

print('儲存檔案')
torch.save(edgeIndexTorch, '../edgeIndex11to13.pt')
np.save('../nodeFeatures11to13.npy', nodeFeatures)
print(f'處理完成，節點數:{numNodes}，邊數:{edgeIndex.shape[1]}')
