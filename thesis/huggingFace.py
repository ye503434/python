from huggingface_hub import HfApi

api = HfApi()

Token = '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'
repoId = 'Hinihao/11-13_DATA'

print("正在上傳")

api.upload_folder(
    folder_path="../",
    repo_id =  repoId ,
    repo_type = "dataset",
    token = Token,

    allow_patterns = [
        "addrMapCombine.pkl",
        "edgeIndex11to13.pt",
        "nodeFeatures11to13.npy",
        "nodeFeaturesFinal.npy"
    ],
    commit_message="備份 11-13 區塊預處理結果"
)
