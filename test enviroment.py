import torch
import torch_scatter
import sys

print(f"--- 環境檢查 ---")
print(f"Python 版本: {sys.version.split()[0]}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 1. 取得顯卡資訊
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 # GB
    reserved_memory = torch.cuda.memory_reserved(0) / 1e9
    allocated_memory = torch.cuda.memory_allocated(0) / 1e9
    free_memory = total_memory - (allocated_memory + reserved_memory)

    print(f"\n--- 顯卡資訊 ---")
    print(f"顯示卡型號: {device_name}")
    print(f"總顯存 (VRAM): {total_memory:.2f} GB")
    print(f"目前剩餘空間: {free_memory:.2f} GB")

    # 2. 測試 torch_scatter 運算 (VGAE 核心運算)
    try:
        src = torch.tensor([1, 2, 3, 4], dtype=torch.float, device='cuda')
        index = torch.tensor([0, 0, 1, 1], device='cuda')
        out = torch_scatter.scatter_sum(src, index)
        print(f"\n--- 運算測試 ---")
        print(f"Scatter 運算測試成功！結果: {out.cpu().numpy()}")
    except Exception as e:
        print(f"\n運算測試失敗: {e}")

else:
    print("\n[警告] 系統抓不到 GPU，請檢查虛擬環境是否正確安裝了 cu124 版本。")
