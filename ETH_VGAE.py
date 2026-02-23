import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

file12to13 = r'D:\12000000to12999999_BlockTransaction.csv'
file11to12 = r'D:\11000000to11999999_BlockTransaction.csv'

# with open('addrMapCombine.pkl','rb') as f:
#     addrToId = pickle.load(f)
# doneMaxId = len(addrToId) #舊的檔案地址筆數
# print(f"目前有{doneMaxId}個地址")
#
# # --把11-13的from到to的地址收集並且寫入二進位檔案。 結果:50196419個地址
# for chunk in tqdm(pd.read_csv(file12to13, chunksize=500000, usecols=['from', 'to']),total = 250):
#     uniqueChunk = set(chunk['from'].unique()) | set(chunk['to'].unique())
#
#     for addr in uniqueChunk:
#         if addr not in addrToId:
#             addrToId[addr] = doneMaxId
#             doneMaxId += 1
#
# with open('addrMapCombine.pkl', 'wb') as f :
#     pickle.dump(addrToId, f)
#
# print(f'掃描完成，總共{len(addrToId)}唯一地址')

# 讀取addr_map.pkl檔案
with open('addrMapCombine.pkl.pkl', 'rb') as f:
    addressToId = pickle.load(f)

num_nodes = len(addressToId)
# 先用三個特徵 入度、出度、總金額 來做陣列
node_features = np.zeros((num_nodes, 3), dtype=np.float32)
#讀取舊的edge_index
old_edge_index = torch.load('edge_index.pt')
edge_list = [old_edge_index.numpy()]#儲存舊的邊
print("轉換數據")
for chunk in tqdm(pd.read_csv(file11to12, chunksize=500000, usecols=['from', 'to', 'value']), total = 240):
    src = chunk['from'].map(addressToId).values
    dst = chunk['to'].map(addressToId).values
    val = chunk['value'].values.astype(np.float32)

    for s, d, v in zip(src, dst, val):
        node_features[s, 0] += 1  # out-degree
        node_features[d, 1] += 1  # in-degree
        node_features[s, 2] += v  # total value

    # 儲存邊
    edges = np.stack([src, dst], axis=0)
    edge_list.append(edges)
#合併所有邊
edge_index = np.concatenate(edge_list, axis=1)
edge_index_torch = torch.from_numpy(edge_index).to(torch.long)

#儲存結果
torch.save(edge_index_torch, 'edge_index.pt')
np.save('node_features.npy' , node_features)
print("處理完成")
