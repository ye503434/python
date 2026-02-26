import numpy as np


nodeFeatures = np.load('../nodeFeatures11to13.npy')#載入特徵檔
nodeFeaturesLog = np.log1p(nodeFeatures)#log1p = log(x+1)

#特徵縮小，Z-Score 標準化：讓數據的平均值為0，標準差為1
mean = nodeFeaturesLog.mean(axis=0)
std = nodeFeaturesLog.std(axis=0)
final = (nodeFeaturesLog - mean) / (std + 1e-8)#std+ 1e-8 防止除以0

np.save('../nodeFeaturesFinal.npy', final.astype(np.float32))

print('處裡完成',f'\n最大原始金額: {np.max(nodeFeatures[:,2])}', f'\n處理後最大值: {np.max(final[:,2])}')
#最大原始金額: 50126404.0
#處理後最大值: 22.896770477294922