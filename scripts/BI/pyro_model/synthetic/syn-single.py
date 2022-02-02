#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append("/users/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/model")

import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer, GCNConv
import pickle
import networkx as nx
from math import floor
from tqdm.autonotebook import tqdm
import seaborn as sns

import sys
sys.path.append("..")

from model.BayesExplainer import BayesExplainer
from model.samplers.NFSampler import NFSampler
from model.samplers.SpikeSlabSampler import SpikeSlabSampler
from model.samplers.RandomWalkSampler import RandomWalkSampler

from IPython.display import set_matplotlib_formats
get_ipython().run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('svg')

G = nx.read_gpickle('data/syn3_G.pickle')
with open('data/syn3_lab.pickle', 'rb') as f:
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
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
# Load everything onto the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
x, edge_index = data.x, data.edge_index

model = Net(x=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best_loss = 100
pbar = tqdm(range(10000))
for epoch in pbar:
    # Training step
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index)
    loss = F.nll_loss(log_logits, data.y)
    loss.backward()
    optimizer.step()

    # Testing step
    model.eval()
    best_loss = loss if loss < best_loss else best_loss
    pbar.set_description("Acc -> %.4f" % torch.mean((torch.argmax(log_logits, dim=1) == data.y).float()).item())


# In[2]:


explainer = GNNExplainer(model, epochs=1000)
node_idx = 565
k = 3
node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)


# In[3]:


k = 3
splines = 8
sampler = NFSampler("syn3_nf_sampler", splines, True, 5, 1.5, device)
explainer = BayesExplainer(model, sampler, node_idx, k, x, data.y, edge_index)
avgs = explainer.train(epochs=2000, lr=0.5, window=500, base="..")
sns.lineplot(x = range(len(avgs)), y = avgs)


# In[4]:


edge_mask = explainer.edge_mask()
ax, G = explainer.visualize_subgraph()
plt.show()


# In[5]:


sns.distplot(explainer.sampler.flow_dist.rsample(torch.Size([10000,])).sigmoid()[:, 7].detach().numpy())


# In[6]:


sampler = SpikeSlabSampler("syn3_spike_slab", 0.25, 1, 5, 10, 1)
explainer = BayesExplainer(model, sampler, node_idx, k, x, data.y, edge_index)
avgs = explainer.train(epochs=10000, lr=0.05, window=500, base="..")
sns.lineplot(x = range(len(avgs)), y = avgs)


# In[7]:


edge_mask = explainer.edge_mask()
ax, G = explainer.visualize_subgraph()
plt.show()


# In[8]:


edge_mask


# In[19]:


sampler = RandomWalkSampler("syn3_random_walk", 0.75)
explainer = BayesExplainer(model, sampler, node_idx, k, x, data.y, edge_index)
avgs = explainer.train(epochs=10000, lr=0.15, window=500, base="..")
sns.lineplot(x = range(len(avgs)), y = avgs)


# In[20]:


edge_mask = explainer.edge_mask()
ax, G = explainer.visualize_subgraph()
plt.show()


# In[21]:


edge_mask


# In[22]:


explainer.sampler.ret_probs(explainer)

