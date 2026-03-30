import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import scipy.stats as stats
from torch_geometric.nn import VGAE, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import umap
import matplotlib.pyplot as plt
import gc
import time


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


    testLoader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=1024,
        shuffle=False
    )

    # 預分配記憶體
    print('正在預分配隱含空間 (Z)...')
    num_nodes = data.num_nodes
    full_z = torch.zeros((num_nodes, channels), dtype=torch.float32)
    recon_errors = torch.zeros(num_nodes, dtype=torch.float32)  # 新增：儲存每個節點的重構損失

    current_idx = 0
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        batch_count = 0
        # --- 第一階段：走訪 DataLoader 抓取原始訊號 ---
        for batch in testLoader:
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index)

            # 計算重構誤差 (Reconstruction Error)
            # 衡量模型對原始邊的重建失敗率，數值越大越異常
            log_prob = model.decoder(z, batch.edge_index, sigmoid=True)
            recon_loss = -torch.log(log_prob + 1e-15)

            # 將邊的損失分配回節點 (使用 scatter_add_ 累加鄰居貢獻)
            row, col = batch.edge_index
            node_recon_loss = torch.zeros(z.size(0), device=device)
            node_recon_loss.scatter_add_(0, row, recon_loss)

            # 儲存結果 (僅取當前 Batch 的中心節點)
            batch_z = z[:batch.batch_size].cpu()
            batch_recon = node_recon_loss[:batch.batch_size].cpu()

            full_z[current_idx: current_idx + batch.batch_size] = batch_z
            recon_errors[current_idx: current_idx + batch.batch_size] = batch_recon

            current_idx += batch.batch_size

        # --- 第二階段：迴圈結束後，進行四階段統計框架運算 ---
        print("正在進行四階段異常偵測框架運算 (回應 Comment 2)...")

        # [步驟 2: 計算行為強度並整合混合得分]
        intensity_scores = torch.norm(full_z, p=2, dim=1).numpy()
        # 混合得分 = 重構誤差 * 向量強度
        combined_scores = intensity_scores * (recon_errors.numpy() + 1e-6)

        # [步驟 3: Modified Z-Score (魯棒性抗噪)]
        median_val = np.median(combined_scores)
        mad_val = np.median(np.abs(combined_scores - median_val))
        # 0.6745 是將 MAD 縮放到與標準差同尺度的常數
        m_z_scores = 0.6745 * (combined_scores - median_val) / (mad_val + 1e-6)

        # [步驟 4: 百分位數校準 (Percentile Calibration)]
        # 將得分轉為 0-100 的排名，解決閾值設定的嚴謹性問題
        percentile_ranks = stats.rankdata(m_z_scores) / len(m_z_scores) * 100

        # --- 新增：分佈感知建模驗證 (Log-Log Plot) ---
        plt.figure(figsize=(8, 6))
        counts, bin_edges = np.histogram(combined_scores, bins=200, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = (counts > 0) & (bin_centers > 0)
        plt.loglog(bin_centers[mask], counts[mask], 'b.', alpha=0.6)
        plt.title("Distribution-Aware: Anomaly Score Power-Law Analysis")
        plt.xlabel("Log(Anomaly Score)")
        plt.ylabel("Log(Probability Density)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig('vgae_distribution_loglog_v5.png', dpi=300)
        print("分佈感知驗證圖表已存檔 (vgae_distribution_loglog_v5.png)。")

        # 標記出極端異常 (Top 0.1%)
        is_extreme = percentile_ranks > 99.9
        print(f"框架運算完成。偵測到 {np.sum(is_extreme)} 個極端異常地址。")

    end_time = time.time()
    total_sec = end_time - start_time
    nodes_per_sec = num_nodes / total_sec

    print("\n" + "=" * 40)
    print("Comment 3: Scalability & Efficiency Report")
    print(f"Total Nodes Processed: {num_nodes:,}")
    print(f"End-to-End Latency: {total_sec:.2f} seconds")
    print(f"Inference Throughput: {nodes_per_sec:,.2f} nodes/sec")
    print(f"Avg Latency per Node: {(1 / nodes_per_sec) * 1000:.6f} ms")
    print("=" * 40)
    # 徹底清空 RAM 以供 UMAP 使用
    print("正在清空記憶體...")
    del data, x, edgeIndex, testLoader
    gc.collect()

    # UMAP 降維
    sample_size = 30000
    print(f"正在執行 UMAP 採樣 (樣本數: {sample_size})...")

    # 隨機採樣
    indices = np.random.choice(full_z.shape[0], sample_size, replace=False)

    # 準備繪圖數據
    z_sample = full_z[indices].detach().numpy()
    # 【關鍵更改】：顏色設定為該樣本的「百分位數排名」，這才能展現異常偵測結果
    sample_ranks = percentile_ranks[indices]
    print("正在鎖定 Top 0.1% 異常數據...")
    top_anomaly_indices = np.where(is_extreme)[0]
    top_scores_extracted = combined_scores[top_anomaly_indices]  # 先存起來
    top_ranks_extracted = percentile_ranks[top_anomaly_indices]  # 先存起來

    # 在執行 UMAP 運算前，清掉不再需要的巨大陣列，節省 RAM
    del combined_scores, m_z_scores, recon_errors
    gc.collect()

    print("正在執行 UMAP 降維運算...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    z_embedding = reducer.fit_transform(z_sample)

    # --- 繪圖邏輯：顏色代表百分位數排名 ---
    fig, ax = plt.subplots(figsize=(12, 8))
    # 顏色映射改為 'YlOrRd' (黃到紅)，紅色越深代表越異常 (Top 0.1%)
    scatter = ax.scatter(z_embedding[:, 0], z_embedding[:, 1],
                         c=sample_ranks, s=8, alpha=0.6, cmap='YlOrRd')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Anomaly Percentile Rank (0-100%)', rotation=270, labelpad=15)

    # 【關鍵更改】：標題要寫出這是「分佈感知評分」，展現專業感
    plt.title(f"UMAP Projection of Ethereum Nodes\n(Color: Percentile Rank based on 4-Stage Scoring)")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")

    # 儲存最終圖表
    plt.savefig('vgae_umap_analysis_v5_final.png', dpi=300)
    print("最終分析圖表已存檔 (vgae_umap_analysis_v5_final.png)。")

    # --- 最後的指標計算 ---
    # print("\n正在計算模型性能指標 (AUC/AP)...")
    # auc, ap = model.test(full_z,
    #                      testData.pos_edge_label_index,
    #                      testData.neg_edge_label_index)

    print("-" * 30)
    # print(f"測試集 AUC: {auc:.4f}")
    # print(f"測試集 AP:  {ap:.4f}")
    print(f"完成時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    # --- 只有在需要輸出名單時才載入 ---
    print("正在提取 Top 0.1% 異常地址清單...")
    address_map = np.load('../nodeAddressList.npy', mmap_mode='r')
    top_anomaly_indices = np.where(is_extreme)[0]
    top_addresses = address_map[top_anomaly_indices]

    # 存成文字檔，這樣你就可以去查這些地址了
    np.savetxt('top_01_percent_anomalies.txt', top_addresses, fmt='%s')
    # --- 關鍵更改：存成更專業的 CSV 格式 ---
    print("正在將 Top 0.1% 名單匯出為 CSV...")

    df_anomaly = pd.DataFrame({
        'Wallet_Address': top_addresses,
        'Anomaly_Score': top_scores_extracted,
        'Percentile_Rank': top_ranks_extracted
    })

    # 依照分數從高到低排序，把最壞的人放在最上面
    df_anomaly = df_anomaly.sort_values(by='Anomaly_Score', ascending=False)
    df_anomaly.to_csv('eth_anomalies_report_v5.csv', index=False)
    print("異常報告已生成")
    print(f"清單已存檔，共計 {len(top_addresses)} 個地址。")
    print("-" * 30)

    # === [步驟一：為 Streamlit 互動頁面儲存 pre-computed 數據] ===
    # 目的：儲存這 30,000 個採樣點的 2D 座標、地址與百分位數排名
    print("\n正在儲存 Streamlit 互動頁面所需的數據...")

    # 確保引用正確 (import scipy.stats as stats)

    # 1. 取得這 採樣點 對應的地址與排名
    z_addresses_sampled = address_map[indices]  # indices 是你原本隨機採樣的 30,000 個索引
    # 從 percentile_ranks 中直接取出對應的排名
    sampled_ranks_final = percentile_ranks[indices]

    # 2. 儲存數據 (這三個變數對應你 umap 的 30,000 點)
    np.save('umap_2d_coords.npy', z_embedding)  # Z_embedding 是 reducer.fit_transform(z_sample) 的結果
    np.save('sampled_addresses.npy', z_addresses_sampled)
    np.save('sampled_percentile_ranks.npy', sampled_ranks_final)

    print("數據已儲存供 Streamlit 使用。")
    print("-" * 30)
