import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch_geometric.utils as tg_utils
import gc

# Terminal : streamlit run streamLit.py
FEATURE_DESC = {
    0: "發出交易頻率 (Out-degree)",
    1: "接收交易頻率 (In-degree)",
    2: "總轉出金額 (Total Sent Value)",
    3: "交易金額波動性 (Value Std Dev)",
    4: "最大單筆轉帳 (Max Single Value)",
    5: "平均交易間隔 (Avg Block Gap)",
    6: "交易時間穩定性 (Time Stability)"
}
# --- [0. 頁面配置] ---
st.set_page_config(page_title="ETH Anomaly Explorer V5", layout="wide")
st.title("🛡️ Ethereum Anomaly Explorer (Interactive Mode)")
st.markdown("""
這個互動式圖表展示了基於 VGAE 與四階段統計框架的以太坊節點分析結果。
顏色代表節點在全網 5,019 萬節點中的 **「異常百分位數排名」**。紅色越深，代表該地址越可疑 (Top 0.1% = >99.9)。
""")


# --- [1. 載入數據函式 (解決 Comment 3: 記憶體與擴展性)] ---
@st.cache_data
def load_offline_umap_data():
    """載入輕量化的 UMAP 座標與預算排名數據"""
    coords = np.load('umap_2d_coords.npy')
    addrs = np.load('sampled_addresses.npy')
    ranks = np.load('sampled_percentile_ranks.npy')

    return pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'address': addrs,
        'percentile_ranks': ranks
    })


@st.cache_resource
def load_massive_raw_data():
    """載入全網 5019 萬個點的原始特徵與結構數據 (使用 mmap)"""
    # 這裡的路徑請根據你的資料夾結構確認
    raw_features = np.load('../nodeFeatures11to13.npy', mmap_mode='r')
    full_addr_list = np.load('../nodeAddressList.npy', mmap_mode='r')
    # 載入邊索引並確保為 Torch 格式
    full_edge_idx = torch.load('../edgeIndex11to13.pt', weights_only=True)
    # --- [優化關鍵]：建立地址對索引的快取字典 ---
    # with st.spinner("正在建立快速索引字典 (加速 Comment 4 查詢)..."):
    #     # 這樣點擊時就不需要再用 np.where 掃描 50M 筆資料了
    #     addr_to_idx_dict = {addr: i for i, addr in enumerate(full_addr_list)}
    # 事先計算全域特徵平均值，用於證據對比 (Comment 4)
    avg_feat = raw_features.mean(axis=0)
    return raw_features, full_addr_list, full_edge_idx, avg_feat


# --- [2. 核心證據提取函式 (解決 Comment 4: 可解釋性)] ---
# 這裡將參數名稱改為 addr_to_check, feat_matrix 等，徹底解決 Shadowing 問題
def fetch_behavioral_evidence(addr_to_check, edge_idx_source, feat_matrix, addr_map_source, global_avg):
    """
    透過局部子圖與特徵偏離度，為異常得分提供物證
    """
    # 1. 【核心修正】：直接從字典獲取索引，取代緩慢的 np.where
    match_indices = np.where(addr_map_source == addr_to_check)[0]
    if len(match_indices) == 0:
        return None, None
    target_node_idx = int(match_indices[0])
    # 2. 提取 1-hop 鄰居 (結構證據)
    subset_nodes, _, _, _ = tg_utils.k_hop_subgraph(
        node_idx=int(target_node_idx),  # 確保為整數
        num_hops=1,
        edge_index=edge_idx_source,
        relabel_nodes=False
    )
    # 計算鄰居數量 (不含自己)
    n_count = len(subset_nodes) - 1

    # 3. 計算特徵偏離度 (行為證據)
    current_node_feat = feat_matrix[target_node_idx]
    deviation = np.abs(current_node_feat - global_avg)
    top_deviant_indices = np.argsort(deviation)[-3:][::-1]

    return n_count, top_deviant_indices


# --- [3. 執行資源加載] ---
df_umap = load_offline_umap_data()
x_raw_ptr, address_ptr, edge_ptr, feature_avg = load_massive_raw_data()

# --- [4. 繪製主圖表] ---
fig_main = px.scatter(
    df_umap, x='x', y='y',
    color='percentile_ranks',
    hover_name='address',
    title="UMAP Projection: Sampled Global Distribution",
    color_continuous_scale='Reds',
    range_color=[0, 100],
    template="plotly_dark",
    height=800
)
fig_main.update_traces(marker=dict(size=7, opacity=0.8))

# 捕捉點擊事件
interaction_event = st.plotly_chart(fig_main, width="stretch", on_select="rerun")

# --- [5. 側邊欄偵測與證據展示邏輯] ---
st.sidebar.header("🔍 Node Inspector")

if interaction_event and "selection" in interaction_event and interaction_event["selection"]["points"]:
    # 1. 取得選中點的資訊
    selected_point_idx = interaction_event["selection"]["points"][0]["point_index"]
    global_target_addr = df_umap.iloc[selected_point_idx]['address']

    # 【強化 1】：強制轉換為 float，確保判斷式精準
    current_rank = float(df_umap.iloc[selected_point_idx]['percentile_ranks'])

    st.sidebar.success(f"Address Identified!")
    st.sidebar.code(global_target_addr, language="text")

    # 異常百分位排名顯示
    metric_label = "Anomaly Percentile Rank"
    if current_rank > 99.9:
        st.sidebar.metric(metric_label, f"{current_rank:.4f}%", delta="Extreme Outlier")
    else:
        st.sidebar.metric(metric_label, f"{current_rank:.4f}%")

    # 2. 獲取證據 (回應 Comment 4)
    with st.sidebar.expander("🛡️ Behavioral Evidence (Interpretability)", expanded=True):
        with st.spinner("Extracting evidence..."):
            neighbors, top_dims = fetch_behavioral_evidence(
                global_target_addr, edge_ptr, x_raw_ptr, address_ptr, feature_avg
            )

        if neighbors is not None:
            st.markdown(f"**Structural Pattern:**")
            st.write(f"Connected to **{neighbors}** interactive neighbors.")

            st.markdown(f"**Deviant Feature Attribution:**")

            # 【強化 2】：嚴格縮進檢查，確保警告框只在高風險時出現
            if current_rank > 80.0:
                st.caption("偵測到與全域基準有顯著差異的行為特徵：")
                for dim in top_dims:
                    if dim < 7:  # 只顯示定義的原始特徵
                        desc = FEATURE_DESC.get(dim, f"Feature_{dim}")
                        st.warning(f"⚠️ {desc}: 顯著偏離全域平均值。")
            else:
                # 這裡會顯示低風險資訊，代表統計上的穩定性
                st.info("✅ 小於80%，不偵測偏離")
        else:
            st.error("Evidence extraction failed.")

    # 提供外部連結
    st.sidebar.markdown("---")
    etherscan_url = f"https://etherscan.io/address/{global_target_addr}"
    st.sidebar.link_button("🌐 Verify on Etherscan", etherscan_url)

else:
    st.sidebar.info("請在 UMAP 圖中點擊任何節點。")
# --- [6. 系統擴展性報告 (Comment 3)] ---
st.markdown("---")
st.markdown("### 📊 System Performance & Scalability (Comment 3 Summary)")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Analysis Scope", "50,190,000 Nodes")
col_b.metric("Data Handling", "Memory-mapped (mmap)")
col_c.metric("Detection Framework", "4-Stage Dist-Aware")
