import os.path as osp
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
from torch_geometric.nn import GNNExplainer, ARMAConv
from torch_geometric.utils import k_hop_subgraph
from math import floor
from scipy import stats

dataset = 'Computers'
path = osp.join('.', 'data', 'Amazon')
dataset = Amazon(path, dataset, transform=T.NormalizeFeatures())
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

def train_model(log=True):
    # k is the number of aggregation hops and x is the hidden feature size
    model = Net(k=3, x=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    line = floor(x.shape[0] // (5/4))
    train_mask = list(range(0, line))
    test_mask = list(range(line, x.shape[0]))

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
    explainer = GNNExplainer(model, epochs=1000, num_hops=1)
    node_idx = 549
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    scores = stats.zscore(edge_mask.cpu().numpy())
    idxs = scores > 10
    print(sum(idxs))
    ax, G = explainer.visualize_subgraph(node_idx, edge_index.T[idxs].T, edge_mask[idxs], y=data.y)
    plt.savefig('explain/node%d.png' % node_idx, dpi=300)