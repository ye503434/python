from time import localtime
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


# 參考比特幣 AML 論文 (Weber et al.) 的 Skip-GCN 概念 [cite: 172, 174]
class SkipSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 第一層：聚合鄰居資訊
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, 2 * out_channels)
        # 輸出層：計算隱含空間分佈
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)
        self.conv_logvar = SAGEConv(2 * out_channels, out_channels)

        # 線性跳躍層：確保原始 7 維特徵能直接影響 64 維隱含空間 [cite: 174]
        self.skip = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 1. 預留原始特徵路徑 [cite: 173]
        x_skip = self.skip(x)

        # 2. GNN 深度聚合
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)

        # 3. 殘差結合：GNN 結構 + 原始特徵，提升對極端值的敏感度 [cite: 174]
        return mu + x_skip, logvar


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("正在執行高效能載入 (含 RAM 優化)...")
    # 這裡的邏輯能確保 32GB RAM 不會溢出
    x_raw = np.load('../nodeFeatures11to13.npy')
    x_log = np.log1p(x_raw)
    del x_raw

    x_final = (x_log - x_log.mean(axis=0)) / (x_log.std(axis=0) + 1e-6)
    del x_log

    x = torch.from_numpy(x_final).float()
    del x_final
    edgeIndex = torch.load('../edgeIndex11to13.pt', weights_only=True)

    data = Data(x=x, edge_index=edgeIndex)

    # --- 超參數優化：64維 + 深度採樣 [15, 10] ---
    channels = 64
    model = VGAE(SkipSAGEEncoder(data.num_features, channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 針對 8GB VRAM 優化 Batch Size
    trainLoader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=1024,  # 如果閃退，請改為 512
        num_workers=1,  # 記憶體高壓下，固定為 1 最安全
        shuffle=True
    )


    def train():
        model.train()
        totalLoss, totalRecon, totalKL = 0, 0, 0
        for batch in trainLoader:
            batch = batch.to(device)
            optimizer.zero_grad()

            z = model.encode(batch.x, batch.edge_index)
            # 在稀疏以太坊網路中加重 ReconLoss 比例 [cite: 137]
            reconLoss = model.recon_loss(z, batch.edge_index)
            klLoss = model.kl_loss() / batch.x.size(0)

            # 訓練策略：稍微偏向重構 (1.2x) 以優化 AUC
            loss = reconLoss * 1.2 + klLoss

            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            totalRecon += reconLoss.item()
            totalKL += klLoss.item()

        return totalLoss / len(trainLoader), totalRecon / len(trainLoader), totalKL / len(trainLoader)


    print(f'啟動優化訓練，目前已突破 0.81 AUC，挑戰 0.83+')
    minLoss = float('inf')
    patience = 5
    button = 0

    for epoch in range(1, 51):
        try:
            loss, r_loss, k_loss = train()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} (Recon: {r_loss:.4f}, KL: {k_loss:.4f})')

            if loss < minLoss:
                minLoss = loss
                button = 0
                torch.save(model.state_dict(), 'vgae_eth_sage_v3_best.pt')
            else:
                button += 1
                if button >= patience:
                    print(f'早停機制觸發，訓練結束。')
                    break
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("!!! GPU 顯存溢出，請手動將 batch_size 減半 !!!")
                exit()
            else:
                raise e

    print(f'結束時間: {localtime().tm_year}, {localtime().tm_hour}:{localtime().tm_min}')