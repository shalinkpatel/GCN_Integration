import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer, GCNConv
import pickle
import networkx as nx

import sys
sys.path.append("..")

from model.BayesExplainer import BayesExplainer
from model.samplers.NFSampler import NFSampler

prefix = '/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/synthetic/'
G = nx.read_gpickle( prefix + 'data/syn3_G.pickle')
with open(prefix + 'data/syn3_lab.pickle', 'rb') as f:
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
    if epoch % 100 == 0:
        print("Acc -> %.4f" % torch.mean((torch.argmax(log_logits, dim=1) == data.y).float()).item())


import numpy as np

with open('/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/PGExplainer/dataset/syn3.pkl', 'rb') as f:
    adj, _, _, _, _, _, _, _, edge_labels = pickle.load(f)
edge_labels = torch.tensor(edge_labels)


from sklearn.metrics import roc_auc_score

auc = 0
auc_gnn_exp = 0
pbar = range(x.shape[0])
done = 0
for n in pbar:
    try:
        k = 3
        splines = 8
        sampler = NFSampler("nf_sampler", len(G.edges), splines, True, 10, 1.5, device)
        explainer = BayesExplainer(model, sampler, n, k, x, edge_index)
        avgs = explainer.train(epochs=3000, lr=0.25, window=500, log=False)
        edge_mask = explainer.edge_mask()
        edges = explainer.edge_index_adj
        labs = edge_labels[explainer.subset, :][:, explainer.subset][edges[0, :], edges[1, :]]
        sub_idx = (labs.long().cpu().detach().numpy() == 1)
        itr_auc = roc_auc_score(labs.long().cpu().detach().numpy()[sub_idx], edge_mask.cpu().detach().numpy()[sub_idx])
        auc += itr_auc
        e_subset = explainer.edge_mask_hard
        explainer = GNNExplainer(model.to(device), epochs=1000, log=False)
        _, edge_mask = explainer.explain_node(n, x.to(device), edge_index.to(device))
        auc_gnn_exp += roc_auc_score(labs.long().cpu().detach().numpy()[sub_idx], edge_mask[e_subset].cpu().detach().numpy()[sub_idx])
        done += 1
        if n % 10 == 0:
            print('EPOCH: %d | AUC: %.3f | AUC GNN_EXP: %.3f | ITR AUC: %.3f' % (n, auc/done, auc_gnn_exp/done, itr_auc))
    except:
        pass

print('FINAL | AUC: %.3f | AUC GNN_EXP: %.3f | ITR AUC: %.3f' % (auc/done, auc_gnn_exp/done, itr_auc))
