import os.path as osp
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GNNExplainer, ARMAConv
from torch_geometric.utils import k_hop_subgraph

dataset = 'Cora'
path = osp.join('.', 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# Define the model
class Net(torch.nn.Module):
    def __init__(self, k=1, x=16):
        super(Net, self).__init__()
        self.conv1 = ARMAConv(dataset.num_features, x)
        self.conv2 = ARMAConv(x, x)
        self.conv3 = ARMAConv(x, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# Load everything onto the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
x, edge_index = data.x, data.edge_index

def train_model():
    # k is the number of aggregation hops and x is the hidden feature size
    model = Net(k=3, x=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_loss = 100
    for epoch in range(1, 501):
        # Training step
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Testing step
        model.eval()
        test_loss = F.nll_loss(log_logits[data.test_mask], data.y[data.test_mask]).item()
        best_loss = test_loss if test_loss < best_loss else best_loss
    return model, x, data.y, edge_index

def extract_subgraph(node_idx, num_hops, edge_index):
    nodes, new_edge_index, mapping, _ = k_hop_subgraph(node_idx, num_hops, edge_index)
    return new_edge_index, node_idx

def run_model(edge_mask, edge_index, model, node_idx):
    edge_index_1 = edge_index[:, torch.tensor(edge_mask).to(device).bool()]
    out = model(x, edge_index_1).detach().cpu()
    return out[node_idx].numpy()