import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
api = HfApi()

Token = os.getenv('HUGGINGFACE_TOKEN')
repoId = os.getenv('HUGGINGFACE_REPO_ID')

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
    ],
    commit_message="備份 11-13 區塊預處理結果"
)
