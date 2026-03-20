import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


#Terminal : streamlit run streamLit.py

# 頁面配置
st.set_page_config(page_title="ETH Anomaly Explorer", layout="wide")
st.title("🛡️ Ethereum Anomaly Explorer (Interactive Mode)")


# 1. 直接讀取你存好的數據
@st.cache_data
def load_offline_data():
    coords = np.load('umap_2d_coords.npy')
    addrs = np.load('sampled_addresses.npy')
    intensity = np.load('sampled_intensity.npy')

    return pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'address': addrs,
        'intensity': intensity
    })


df = load_offline_data()

# 2. 側邊欄：顯示詳細資訊
st.sidebar.header("🔍 Node Inspector")
st.sidebar.write("在圖中點擊一個節點，即可獲取其 Etherscan 連結。")

# 3. 繪圖
fig = px.scatter(
    df, x='x', y='y',
    color='intensity',
    hover_name='address',
    title="UMAP Projection of 50M Nodes (Sampled)",
    color_continuous_scale='Viridis',
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
    target_score = df.iloc[point_index]['intensity']

    # 在側邊欄顯示結果
    st.sidebar.success(f"Selected Address Identified!")
    st.sidebar.code(target_addr, language="text")
    st.sidebar.metric("Latent Intensity", f"{target_score:.4f}")

    # Etherscan 跳轉按鈕
    url = f"https://etherscan.io/address/{target_addr}"
    st.sidebar.link_button("🌐 Open in Etherscan", url)

    # 提示訊息
    st.sidebar.info("Check for: Automated scripts, phishing clusters, or high-frequency internal transactions.")
