import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


#Terminal : streamlit run streamLit.py

# 頁面配置
st.set_page_config(page_title="ETH Anomaly Explorer", layout="wide")
st.title("🛡️ Ethereum Anomaly Explorer (Interactive Mode)")
st.markdown("""
這個互動式圖表展示了基於 VGAE 與四階段統計框架的以太坊節點分析結果。
顏色代表節點在全網 5,019 萬節點中的 **「異常百分位數排名」**。紅色越深，代表該地址越可疑 (Top 0.1% = >99.9)。
""")

# 1. 直接讀取你存好的數據
@st.cache_data
def load_offline_data():
    coords = np.load('umap_2d_coords.npy')
    addrs = np.load('sampled_addresses.npy')
    percentile_ranks = np.load('sampled_percentile_ranks.npy')

    return pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'address': addrs,
        'percentile_ranks': percentile_ranks
    })



df = load_offline_data()

# 2. 側邊欄：顯示詳細資訊
st.sidebar.header("🔍 Node Inspector")
st.sidebar.write("在圖中點擊一個節點，即可獲取其 Etherscan 連結。")

# 3. 繪圖
fig = px.scatter(
    df, x='x', y='y',
    color='percentile_ranks',
    hover_name='address',
    title="UMAP Sampled 30000 Nodes",
    color_continuous_scale='Reds',
    range_color=[0, 100],
    template="plotly_dark",
    height=800
)

# 優化顯示效果
fig.update_traces(marker=dict(size=6, opacity=0.8))

# 4. 捕捉點擊事件
# 使用 on_select="rerun" 來即時獲取點擊資訊
selected_data = st.plotly_chart(fig, width="stretch", on_select="rerun")

# 5. 跳轉邏輯
if selected_data and "selection" in selected_data and selected_data["selection"]["points"]:
    # 獲取點擊到的點在 DataFrame 中的索引
    point_index = selected_data["selection"]["points"][0]["point_index"]
    target_addr = df.iloc[point_index]['address']
    target_score = df.iloc[point_index]['percentile_ranks']

    # 在側邊欄顯示結果
    st.sidebar.success(f"Selected Address Identified!")
    st.sidebar.code(target_addr, language="text")
    label = "Anomaly Percentile Rank (0-100%)"
    if target_score > 99.9:
        st.sidebar.metric(label, f"{target_score:.4f}%", delta="Top 0.1% Extreme Anomaly")
    else:
        st.sidebar.metric(label, f"{target_score:.4f}%")
    # Etherscan 跳轉按鈕
    url = f"https://etherscan.io/address/{target_addr}"
    st.sidebar.link_button("🌐 Open in Etherscan", url)

    # 提示訊息
    st.sidebar.info("Check for: tornado_cash_user, phishing_cluster, multi_level_transfer.")

# 顯示統計資料 (簡單的回應 Comment 2 與 Comment 3)
st.markdown("### 📊 Dataset Summary (50.19M Nodes)")
col1, col2 = st.columns(2)
col1.metric("Sampled Interaction Nodes", f"{len(df):,}", "UMAP Coordinates Pre-computed")
total_nodes_50m = 50190000
top_01_percent_nodes = int(total_nodes_50m * 0.001)
col2.metric("Identified Top 0.1% Anomaly", f"{top_01_percent_nodes:,}", "Distribution-Aware Threshold")