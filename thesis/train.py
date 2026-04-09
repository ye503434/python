import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 新增：自動降速工具


# --- 1. 升級版 Encoder：改用 ELU 激活函數 ---
class SkipSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, 2 * out_channels)
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)
        self.conv_logvar = SAGEConv(2 * out_channels, out_channels)
        self.skip = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x_skip = self.skip(x)
        # 使用 ELU 替代 ReLU，避免 64 維空間中的神經元死亡問題
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))

        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu + x_skip, logvar


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("正在執行高效能載入 (含 RAM 優化)...")
    x_raw = np.load('../nodeFeatures11to13.npy')
    x_log = np.log1p(x_raw)
    del x_raw
    x_final = (x_log - x_log.mean(axis=0)) / (x_log.std(axis=0) + 1e-6)
    del x_log
    x = torch.from_numpy(x_final).float()
    del x_final
    edgeIndex = torch.load('../edgeIndex11to13.pt', weights_only=True)

    data = Data(x=x, edge_index=edgeIndex)

    # --- 2. 參數微調：LR=0.0005 並加入 L2 權重衰減 ---
    channels = 64
    model = VGAE(SkipSAGEEncoder(data.num_features, channels)).to(device)
    # weight_decay 能讓權重更穩定，避免在 0.76 附近震盪
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    # 當 Recon Loss 連續 3 圈沒進步，將學習率自動減半
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # --- 3. 採樣廣度優化 [20, 15] ---
    trainLoader = NeighborLoader(
        data,
        num_neighbors=[20, 15],  # 擴大視野，捕捉更深層的洗錢鏈
        batch_size=1024,
        num_workers=1,
        shuffle=True
    )

    ANNEAL_EPOCHS = 50  # 放慢退火速度，讓模型有 50 圈時間適應


    def train(epoch):
        model.train()
        totalLoss, totalRecon, totalKL = 0, 0, 0
        # --- 4. KL 權重限制：最高只到 0.2，保留 80% 的精力專攻 Recon ---
        klWeight = min(0.2, epoch / ANNEAL_EPOCHS)

        for batch in trainLoader:
            batch = batch.to(device)
            optimizer.zero_grad()

            z = model.encode(batch.x, batch.edge_index)
            reconLoss = model.recon_loss(z, batch.edge_index)
            klLoss = model.kl_loss() / batch.x.size(0)

            # 複合損失函數
            loss = reconLoss * 1.2 + klLoss * klWeight

            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            totalRecon += reconLoss.item()
            totalKL += klLoss.item()

        return totalLoss / len(trainLoader), totalRecon / len(trainLoader), totalKL / len(trainLoader), klWeight


    print(f'啟動 Pro 版優化訓練，挑戰 0.85 AUC 紀錄')
    minRecon = float('inf')
    patience = 15  # 增加耐心到 15 圈，配合 Scheduler 進行二段式衝刺
    button = 0

    for epoch in range(1, 151):  # 給予更多訓練空間
        try:
            loss, r_loss, k_loss, w_kl = train(epoch)
            # 更新學習率排程器
            scheduler.step(r_loss)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} (Recon: {r_loss:.4f}, KL: {k_loss:.4f}, Weight:{w_kl:.2f})')

            if r_loss < minRecon:
                minRecon = r_loss
                button = 0
                torch.save(model.state_dict(), '../vgae_eth_sage_v5_pro.pt')
            else:
                button += 1
                if button >= patience:
                    print(f'突破失敗，Recon Loss 已連續 {patience} 圈未創紀錄。')
                    break
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("!!! GPU 顯存溢出，請手動將 batch_size 降至 512 !!!")
                exit()
            else:
                raise e

    print(f'最佳 Recon Loss: {minRecon:.4f}')
    print(time.strftime("%Y-%M-%D %H:%M:%S", time.localtime()))
