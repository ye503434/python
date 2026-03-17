from time import localtime
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import VGAE, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import umap
import matplotlib.pyplot as plt
import gc


# F.relu 改 F.elu
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
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu + x_skip, logvar


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("正在載入數據...")
    x_raw = np.load('../nodeFeatures11to13.npy')
    x_log = np.log1p(x_raw)
    del x_raw
    x_final = (x_log - x_log.mean(axis=0)) / (x_log.std(axis=0) + 1e-6)
    del x_log
    x = torch.from_numpy(x_final).float()
    del x_final
    edgeIndex = torch.load('../edgeIndex11to13.pt', weights_only=True)

    data = Data(x=x, edge_index=edgeIndex)

    # 參數與權重載入
    channels = 64
    model = VGAE(SkipSAGEEncoder(data.num_features, channels)).to(device)
    model.load_state_dict(torch.load('vgae_eth_sage_v5_pro.pt'))
    model.eval()

    transform = T.RandomLinkSplit(
        num_val=0, num_test=0.01,
        is_undirected=False, add_negative_train_samples=False, split_labels=True
    )
    _, _, testData = transform(data)

    testLoader = NeighborLoader(
        testData,
        num_neighbors=[10, 5],
        batch_size=1024,
        shuffle=False
    )

    # 預分配記憶體
    print('正在預分配隱含空間 (Z)...')
    num_nodes = data.num_nodes
    full_z = torch.zeros((num_nodes, channels), dtype=torch.float32)

    current_idx = 0
    with torch.no_grad():
        for batch in testLoader:
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index)
            batch_z = z[:batch.batch_size].cpu()
            full_z[current_idx: current_idx + batch.batch_size] = batch_z
            current_idx += batch.batch_size

    #徹底清空 RAM 以供 UMAP 使用
    print("正在清空記憶體...")
    testData.x = None
    del data, x, edgeIndex, testLoader
    gc.collect()

    #UMAP 降維
    sample_size = 30000
    print(f"正在執行 UMAP (樣本數: {sample_size})...")
    indices = np.random.choice(full_z.shape[0], sample_size, replace=False)
    z_sample = full_z[indices].detach().numpy()

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    z_embedding = reducer.fit_transform(z_sample)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(z_embedding[:, 0], z_embedding[:, 1],
                          c=z_sample[:, 0], s=1, alpha=0.5, cmap='viridis')
    plt.colorbar(scatter, label='Latent Intensity')
    plt.title(f"UMAP Projection (v5-Pro Final Study)")
    plt.savefig('vgae_umap_analysis_v5.png', dpi=300)
    print("圖表已存檔。")

    print("正在計算最終 AUC/AP...")
    auc, ap = model.test(full_z,
                         testData.pos_edge_label_index,
                         testData.neg_edge_label_index)
    print(f"AUC: {auc:.4f}")
    print(f"AP:  {ap:.4f}")