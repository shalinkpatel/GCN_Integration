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
from tqdm import tqdm
import seaborn as sns
from scipy.sparse import coo_matrix,csr_matrix

import sys
sys.path.append("..")

from BayesianExplainer import BayesianExplainer

from IPython.display import set_matplotlib_formats

prefix = '/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/synthetic/'
G = nx.read_gpickle( prefix + 'data/syn1_G.pickle')
with open(prefix + 'data/syn1_lab.pickle', 'rb') as f:
    labels = pickle.load(f)

x = torch.tensor([x[1]['feat'] for x in G.nodes(data=True)])
edge_index = torch.tensor([x for x in G.edges])
edge_index_flipped = edge_index[:, [1, 0]]
edge_index = torch.cat((edge_index, edge_index_flipped))
y = torch.tensor(labels, dtype=torch.long)
data = Data(x=x, edge_index=edge_index.T, y=y)

class Net(torch.nn.Module):
    def __init__(self, x=64):
        super(Net, self).__init__()
        self.conv1 = GCNConv(10, x)
        self.conv2 = GCNConv(x, x)
        self.conv3 = GCNConv(x, x)
        self.fc = torch.nn.Linear(x, max(y).tolist()+1)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        return self.fc(x)
    
# Load everything onto the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
x, edge_index = data.x, data.edge_index

model = Net(x=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best_loss = 100
pbar = range(10000)
for epoch in pbar:
    # Training step
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index)
    loss = F.cross_entropy(log_logits, data.y)
    loss.backward()
    optimizer.step()

    # Testing step
    model.eval()
    best_loss = loss if loss < best_loss else best_loss
    #pbar.set_description("Acc -> %.4f" % torch.mean((torch.argmax(log_logits, dim=1) == data.y).float()).item())

import numpy as np

with open('/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/PGExplainer/dataset/syn1.pkl', 'rb') as f:
    adj, _, _, _, _, _, _, _, edge_labels = pickle.load(f)
edge_labels = torch.tensor(edge_labels)

auc = 0
for n in tqdm(range(x.shape[0])):
    k = 3
    sharp = 1e-4
    explainer = BayesianExplainer(model, n, k, x, edge_index, sharp)
    avgs = explainer.train(epochs=5000, lr=0.05, window=500, log=False)
    edge_mask = explainer.edge_mask()
    edges = explainer.edge_index_adj
    labs = edge_labels[explainer.subset, :][:, explainer.subset][edges[0, :], edges[1, :]]
    auc += (edge_mask[labs.long() == 1].sum() + (1 - edge_mask[labs.long() == 0]).sum())/edge_mask.shape[0]
print(auc/x.shape[0])
