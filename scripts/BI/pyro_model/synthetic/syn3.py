import os.path as osp
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer, GCNConv
from torch_geometric.utils import k_hop_subgraph, from_networkx
import pickle
import networkx as nx
from math import floor

prefix = '/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/examples/syn/'
G = nx.read_gpickle( prefix + 'data/syn4_G.pickle')
with open(prefix + 'data/syn4_lab.pickle', 'rb') as f:
    labels = pickle.load(f)

x = torch.tensor([x[1]['feat'] for x in G.nodes(data=True)])
edge_index = torch.tensor([x for x in G.edges])
edge_index_flipped = edge_index[:, [1, 0]]
edge_index = torch.cat((edge_index, edge_index_flipped))
y = torch.tensor(labels, dtype=torch.long)
data = Data(x=x, edge_index=edge_index.T, y=y)

class Net(torch.nn.Module):
    def __init__(self, k=1, x=64):
        super(Net, self).__init__()
        self.conv1 = GCNConv(10, x)
        self.conv2 = GCNConv(x, x)
        self.conv3 = GCNConv(x, max(y).tolist()+1)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
# Load everything onto the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
x, edge_index = data.x, data.edge_index

def train_model(log=False):
    # k is the number of aggregation hops and x is the hidden feature size
    model = Net(k=3, x=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    line = floor(x.shape[0] // (5/4))
    train_mask = list(range(0, line))
    test_mask = list(range(line, x.shape[0]))

    best_loss = 100
    for epoch in range(1, 10001):
        # Training step
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.cross_entropy(log_logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Testing step
        model.eval()
        test_loss = F.cross_entropy(log_logits[test_mask], data.y[test_mask]).item()
        best_loss = test_loss if test_loss < best_loss else best_loss
        if log and epoch % 50 == 1:
            print(test_loss)
            print(torch.mean(torch.tensor(torch.argmax(log_logits, dim=1) == data.y, dtype=torch.float32)))
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
    explainer = GNNExplainer(model, epochs=1000)
    node_idx = 549
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
    plt.savefig('explain/node%d.png' % node_idx, dpi=300)