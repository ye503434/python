import torch
import torch_scatter

from torch_geometric.data import Data

edge_index = torch.tensor([[0,1,1,2],
                           [1,0,2,1]], dtype = torch.long)
x = torch.tensor([[-1],[0],[1]], dtype = torch.float)

data = Data(x=x, edge_index= edge_index)
print(data)
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Scatter: {torch_scatter.__version__}")
