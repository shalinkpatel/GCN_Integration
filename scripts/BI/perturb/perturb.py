import os.path as osp
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GNNExplainer, ARMAConv
from torch_geometric.utils import k_hop_subgraph, from_networkx
import networkx as nx
import json
from math import floor
import random
import numpy as np

g = nx.readwrite.gml.read_gml("scripts/BI/perturb/data/syn_graph_4.gml", label=None)
with open('scripts/BI/perturb/data/syn_graph_labels_4.json', 'r') as f:
    labels = json.load(f) 
labels = torch.tensor([labels[str(i)] for i in range(1, len(labels.keys()) + 1)])
x = torch.rand((labels.shape[0], 10), dtype=torch.float32)
x[:(labels.shape[0] - sum(labels.numpy().tolist())), :] = 0
data = from_networkx(g)
data.x = x
data.y = labels

# Define the model
class Net(torch.nn.Module):
    def __init__(self, k=1, x=64):
        super(Net, self).__init__()
        self.conv1 = ARMAConv(10, x)
        self.conv2 = ARMAConv(x, x)
        self.conv3 = ARMAConv(x, 2)

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

def train_model(log=False):
    # k is the number of aggregation hops and x is the hidden feature size
    model = Net(k=3, x=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    idxs = list(range(x.shape[0]))
    random.shuffle(idxs)
    train_mask = idxs[:(4 * x.shape[0] // 5)]
    test_mask = idxs[(4 * x.shape[0] // 5):]

    best_loss = 100
    for epoch in range(1, 501):
        # Training step
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.nll_loss(log_logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Testing step
        model.eval()
        test_loss = F.nll_loss(log_logits[test_mask], data.y[test_mask]).item()
        best_loss = test_loss if test_loss < best_loss else best_loss
        if log and epoch % 50 == 1:
            print(best_loss)
    return model, x, data.y, edge_index

def extract_subgraph(node_idx, num_hops, edge_index):
    nodes, new_edge_index, mapping, _ = k_hop_subgraph(node_idx, num_hops, edge_index)
    return new_edge_index, node_idx

def run_model(edge_mask, edge_index, model, node_idx):
    edge_index_1 = edge_index[:, torch.tensor(edge_mask).to(device).bool()]
    out = model(x, edge_index_1).detach().cpu()
    return out[node_idx].numpy()

if __name__ == '__main__':
    model, x, data.y, edge_index = train_model(log=True)
    explainer = GNNExplainer(model, epochs=1000, num_hops=2)
    node_idx = 1
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
    plt.savefig('scripts/BI/perturb/explain/node%d.png' % node_idx, dpi=300)