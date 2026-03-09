from time import localtime
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import VGAE, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


class SAGEEncoder(torch.nn.Module):  # GCN是無向的，SAGE是有向的 GAT偏向大額交易適合金融
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, 2 * out_channels)
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)
        self.conv_logvar = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_raw = np.load('../nodeFeatures11to13.npy')

    x_log = np.log1p(x_raw)
    x_final = (x_log - x_log.mean(axis=0)) / (x_log.std(axis=0) + 1e-6)

    x = torch.from_numpy(x_final).float()
    edgeIndex = torch.load('../edgeIndex11to13.pt', weights_only=True)

    # Pyg的Data物件
    data = Data(x=x, edge_index=edgeIndex)

    channels = 32
    model = VGAE(SAGEEncoder(data.num_features, channels)).to(device)
    model.load_state_dict(torch.load('vgae_eth_sage.pt'))
    model.eval()

    transform = T.RandomLinkSplit(
        num_val= 0 , num_test= 0.01,
        is_undirected=False,
        add_negative_train_samples=False,
        split_labels = True
    )
    _, _, testData = transform(data)

    testLoader = NeighborLoader(
        testData,
        num_neighbors=[10, 5],
        batch_size=4096,
        shuffle=False
    )

    print('開始計算地址隱藏特徵')
    zlist = []
    with torch.no_grad():
        for batch in testLoader :
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index)
            zlist.append(z[:batch.batch_size].cpu())

    full_z = torch.cat(zlist, dim=0)
    print("正在計算 AUC 與 AP 分數...")
    auc, ap = model.test(full_z,
                         testData.pos_edge_label_index,
                         testData.neg_edge_label_index)
    #計算 AUC 與 AP 分數
    print(f"AUC: {auc:.4f} ")#越接近1分類越準
    print(f"AP:  {ap:.4f}  ")#平均精準度
    time = localtime()
    print(f'結束時間 {time.tm_year},{time.tm_hour}:{time.tm_min}:{time.tm_sec}')
