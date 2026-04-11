import os

import numpy as np
import pandas as pd
import pickle
import torch
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()
file12to13 = os.getenv('FILE12TO13')
file11to12 = os.getenv('FILE11TO12')
chunkSize = 500000

# 把11-13的from到to的地址收集並且寫入二進位檔案。 結果: 共有50196418個唯一節點，邊數:382171280
uniqueAddr = set()

# for filePath in [file11to12,file12to13]:
#     print('掃描中......')
#     total = 361 if "11000000" in filePath else 405
#     for chunk in tqdm(pd.read_csv(filePath, chunksize=chunkSize, usecols=['from', 'to']), total= total):
#         chunk = chunk.replace(r'^\s*$',np.nan,regex=True).dropna(subset = ['from','to'])
#         uniqueAddr.update(chunk['from'].str.lower().str.strip().unique())
#         uniqueAddr.update(chunk['to'].str.lower().str.strip().unique())
#
# addrToId = {addr: i for i , addr in enumerate(uniqueAddr)}
#
# with open('../addrMapCombine.pkl', 'wb') as f :
#     pickle.dump(addrToId, f)
# numNodes = len(addrToId)
# print(f'掃描完成，總共{numNodes}唯一地址')

with open('../addrMapCombine.pkl', 'rb') as f:
    addrToId = pickle.load(f)

numNodes = len(addrToId)
print('初始化特徵矩陣和邊')
nodeFeatures = np.zeros((numNodes, 7), dtype=np.float32)
lastBlock = np.zeros(numNodes, dtype=np.int64)
edgeList = []

for filePath in [file11to12, file12to13]:
    print(f'將數據轉換為ID格式:{filePath}')
    total = 361 if "11000000" in filePath else 405

    for chunk in tqdm(pd.read_csv(filePath, chunksize=chunkSize, usecols=['from', 'to', 'value','blockNumber']), total=total):
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
            print(f'\n偵測到未知地址，請檢查 addrMap')
            exit()

        #數值處理 Wei 轉 Ether 1e18 = 10的18次方
        #.values可以改成 .to_numpy() 更符合現代
        val = chunk['value'].values.astype(np.float32) / 1e18
        blocks = chunk['blockNumber'].values.astype(np.int64)
        for s, d, v, b in zip(src, dst, val, blocks):
            nodeFeatures[s, 0] += 1  # 出度
            nodeFeatures[d, 1] += 1  # 入度
            nodeFeatures[s, 2] += v  # 總交易金額
            nodeFeatures[s, 3] += v**2 # 金額平方和
            nodeFeatures[s, 4] = max(nodeFeatures[s, 4],v) # 最大單筆金額

            if lastBlock[s] > 0 :
                gap = float(b - lastBlock[s])
                if gap>=0:
                    nodeFeatures[s,5] += gap   #累加區塊差
                    nodeFeatures[s,6] += gap**2#累加去塊差平方
            lastBlock[s] = b

        edgeList.append(np.stack([src, dst], axis=0))
#釋放輔助陣列，節省記憶體
del lastBlock
print("\n正在計算特徵後處理 (標準差)...")
# 有效交易次數 (計算平均值與標準差時的分母)
# 區塊間隔數為 出度 - 1
out_counts = nodeFeatures[:, 0].copy()
gap_counts = np.maximum(out_counts - 1, 1e-6) # 防止除以 0
out_counts[out_counts == 0] = 1

# 1. 計算金額標準差 (Feature 3)
mean_sq_val = nodeFeatures[:, 3] / out_counts
mean_val = nodeFeatures[:, 2] / out_counts
nodeFeatures[:, 3] = np.sqrt(np.maximum(mean_sq_val - mean_val**2, 0))

# 2. 計算平均區塊差 (Feature 5)
nodeFeatures[:, 5] = nodeFeatures[:, 5] / gap_counts

# 3. 計算區塊差標準差 (Feature 6: 時間穩定性) [cite: 416, 437]
mean_sq_gap = nodeFeatures[:, 6] / gap_counts
mean_gap = nodeFeatures[:, 5] # 已經是平均值了
nodeFeatures[:, 6] = np.sqrt(np.maximum(mean_sq_gap - mean_gap**2, 0))

print("\n正在合併邊...")
edgeIndex = np.concatenate(edgeList, axis=1)
edgeIndexTorch = torch.from_numpy(edgeIndex).to(torch.long)

print('儲存 7 維特徵與邊...')
torch.save(edgeIndexTorch, '../edgeIndex11to13.pt')
np.save('../nodeFeatures11to13.npy', nodeFeatures)
print(f'處理完成！節點數: {numNodes}, 邊數: {edgeIndex.shape[1]}, 特徵維度: {nodeFeatures.shape[1]}')