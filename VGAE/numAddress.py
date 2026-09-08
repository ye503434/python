import pickle
import numpy as np

# 1. 載入當初存好的地址對應字典
print("正在載入 addrMapCombine.pkl (這可能需要一點時間)...")
with open('../addrMapCombine.pkl', 'rb') as f:
    addrToId = pickle.load(f)

numNodes = len(addrToId)
print(f"載入完成，共有 {numNodes} 個唯一地址。")

# 2. 建立地址陣列 (ID -> Address)
print("正在進行 ID 轉地址映射...")
# 先建立一個空陣列，dtype=object 可以存放字串
# 注意：5019 萬個地址約需 2-4GB RAM，你的 32GB RAM 非常夠用
nodeAddresses = [None] * numNodes

for addr, idx in addrToId.items():
    nodeAddresses[idx] = addr

# 3. 轉成 numpy array 並儲存
print("正在儲存為 ../nodeAddressList.npy ...")
nodeAddresses = np.array(nodeAddresses)
np.save('../nodeAddressList.npy', nodeAddresses)
