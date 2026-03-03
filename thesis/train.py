from time import localtime
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


class SAGEEncoder(torch.nn.Module):  # GCN是無向的，SAGE是有向的 GAT偏向大額交易適合金融
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)  # 看直接交易的鄰居
        self.conv2 = SAGEConv(2 * out_channels, 2 * out_channels)  # 看鄰居的鄰居
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)  # 均值
        self.conv_logvar = SAGEConv(2 * out_channels, out_channels)  # 方差

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


if __name__ == '__main__':
    x = torch.from_numpy(np.load('../nodeFeaturesFinal.npy')).float()
    edgeIndex = torch.load('../edgeIndex11to13.pt', weights_only=True)

    # Pyg的Data物件
    data = Data(x=x, edge_index=edgeIndex)
    numNodes = data.num_nodes

    channels = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGAE(SAGEEncoder(data.num_features, channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 因為顯存 8GB ，所以batch_size設在512到2048
    trainLoader = NeighborLoader(
        data,
        num_neighbors=[10, 5],  # 第一層抽取15個，第二成10個
        batch_size=2048,
        num_workers=1,
        shuffle=True
    )


    def train():
        model.train()
        totalLoss = 0
        totalRecon = 0
        totalKL = 0
        for batch in trainLoader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model.encode(batch.x, batch.edge_index)

            reconLoss = model.recon_loss(z, batch.edge_index)
            klLoss = model.kl_loss() / batch.x.size(0)
            loss = reconLoss + klLoss

            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            totalRecon += reconLoss.item()
            totalKL += klLoss.item()

        return totalLoss / len(trainLoader), totalRecon / len(trainLoader), totalKL / len(trainLoader)


    print(f'開始在{device}訓練11-13區塊數據')
    minLoss = float('inf')
    patience = 5
    button = 0
    for epoch in range(1, 51):
        loss, r_loss, k_loss = train()
        print(f'Epoch: {epoch:03d}, Total: {loss:.4f} (Recon: {r_loss:.4f}, KL: {k_loss:.4f})')
        if loss < minLoss :
            minLoss = loss
            button = 0
            torch.save(model.state_dict(), f'vgae_eth_sage_epoch_best.pt')
        else:
            button +=1
            print(f'觸發{button}')
        if  button >= patience:
            print(f'在{epoch}停止')
            break
        if epoch % 10 == 0 :
            torch.save(model.state_dict(), f'vgae_eth_sage_epoch_{epoch}.pt')

    # 儲存訓練好的模型 SAGE 大腦
    time = localtime()
    print(f'結束時間 {time.tm_year},{time.tm_hour}:{time.tm_min}:{time.tm_sec}')

    torch.save(model.state_dict(), 'vgae_eth_sage.pt')
