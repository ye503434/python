import torch
import torch.nn.functional as F
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv , VGAE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges

chunkSize = 100000

dataset = Planetoid(root='/tmp/Cora' , name='Cora')
data = dataset[0]
print(data)

transform = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    split_labels=True,
    add_negative_train_samples=False
)
train_data, val_data, test_data = transform(data)

class GCNEncoder(torch.nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels , 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels , out_channels)
        self.conv_logvar = GCNConv(2 * out_channels , out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x , edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

channels = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGAE(GCNEncoder(dataset.num_features, channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1 / train_data.numNodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range (1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

model.eval()
with torch.no_grad():
    z = model.encode(test_data.x, test_data.edge_index)
    auc, ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    print(f'AUC: {auc:.4f}, AP: {ap:.4f}')