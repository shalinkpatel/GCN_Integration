# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cell_line', nargs=1, type=str, help='cell line to run on')
parser.add_argument('--name', nargs=1, type=str, help='name of dataset')
parser.add_argument('--shuffle', nargs=1, type=str, help='Permute nodes or not')
args = parser.parse_args()

cl = args.cell_line[0]

# %%
import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import shap

# %%
from dgl.nn.pytorch import SAGEConv, GraphConv, ChebConv, TAGConv, GATConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='pool', activation=torch.tanh)
        self.conv2 = ChebConv(hidden_size, hidden_size1, 4)
        self.conv3 = TAGConv(hidden_size1, hidden_size2, activation=F.leaky_relu, k=3)
        self.conv4 = SAGEConv(hidden_size2, hidden_size3, aggregator_type='pool', activation=torch.tanh)
        self.conv5 = ChebConv(hidden_size3, hidden_size4, 4)
        self.conv6 = TAGConv(hidden_size4, num_classes, activation=F.leaky_relu, k=3)
        x = 150
        self.encoder = nn.Sequential(
            nn.Conv2d(1, x, (3, 3)),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(x, 2*x, (3, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(2*x, 1, (3, 2)),
        )

    def forward(self, g, inputs):
        h = self.encoder(inputs).reshape(-1, 94)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv1(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv3(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv4(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv5(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv6(g, h)
        h = torch.sigmoid(h)
        return h

# %%
model = GCN(94, 1350, 750, 250, 100, 25, 2)
model.load_state_dict(torch.load("/gpfs_home/spate116/data/spate116/GCN/%s/res/best_run.model" % cl))

# %%
import pickle
with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_class1_unflattened.pickle' % cl, 'rb') as f:
    data = pickle.load(f)
    data_embedding = data.x.reshape(data.x.shape[0], 1, data.x.shape[1], data.x.shape[2]).float()

# %%
import networkx as nx
import sklearn.preprocessing as preprocessing
import dgl

edges = data.edge_index.t()
adj = list(map(lambda x: (x[0].item(), x[1].item()), edges))

graph = nx.read_gpickle("/gpfs_home/spate116/data/spate116/GCN/%s/data/graph.pickle" % cl)
weights = [x[2] for x in graph.edges.data('weight')]
robust_scaler = preprocessing.RobustScaler()
weights = np.ndarray.flatten(robust_scaler.fit_transform(np.array(weights).reshape(-1, 1)))

y = torch.tensor(list(map(lambda x: x[0], data.y)), dtype=torch.long)

G = dgl.DGLGraph(adj)

G.ndata['feat'] = data_embedding
G.ndata['expr'] = y
G.edata['weight'] = torch.tensor(weights, dtype=torch.float)

# %%
from tqdm import tqdm
class GCN_N(nn.Module):
    def __init__(self, m, X, G):
        super(GCN_N, self).__init__()
        self.m = m
        self.X = X
        self.G = G
        
    def forward(self, n_mask, inputs):
        return self.m(self.G, inputs.reshape(data.x.shape[0], 1, data.x.shape[1], data.x.shape[2]))[n_mask.long()].reshape(1, -1)

# %%
m = GCN_N(model, data_embedding, G)
X = torch.zeros(1, data.x.shape[0], 1, data.x.shape[1], data.x.shape[2]).float()

m = m.cuda()
X = X.cuda()
e = shap.DeepExplainer(m, [torch.tensor(0), X])
print('Finished Definition')

# %%
shap_values = e.shap_values([torch.tensor(1), data_embedding.reshape(1, data.x.shape[0], 1, data.x.shape[1], data.x.shape[2])])
print(shap_values)

# %%
with open('/gpfs_home/spate116/data/spate116/GCN/%s/res/shapley_graph.res' % cl, 'wb') as f:
    pickle.dump({1: shap_values, 2: list(range(50, 60))}, f)

print('Done :)')
