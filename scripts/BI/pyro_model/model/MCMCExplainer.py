from copy import copy
from os.path import exists

from math import sqrt

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
import networkx as nx
import pyro
import torch
from torch.utils.tensorboard import SummaryWriter
from pyro.optim import Adam
from pyro.infer import SVI
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from tqdm.autonotebook import tqdm

from samplers.BaseSampler import BaseSampler

from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc import MCMC


class MCMCExplainer:
    def __init__(self, model: torch.nn.Module, sampler: BaseSampler, node_idx: int, k: int,
                 x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor):
        device = torch.device('cpu')
        self.model = model.to(device)
        self.x = x.to(device)
        self.y_true = y.to(device)
        self.edge_index = edge_index.to(device)
        self.node_idx = node_idx
        self.k = k
        self.subset, self.edge_index_adj, self.mapping, self.edge_mask_hard = k_hop_subgraph(
            self.node_idx, k, self.edge_index, relabel_nodes=True)
        self.total_mapping = {k: i for k, i in enumerate(self.subset.tolist())}
        self.x_adj = self.x[self.subset]
        self.device = device

        with torch.no_grad():
            self.preds = model(self.x_adj, self.edge_index_adj).exp()

        self.N = self.edge_index_adj.size(1)

        self.sampler = sampler

    def train(self, samples: int = 1000, **kwargs):
        pyro.get_param_store().clear()

        sample_model = lambda x: self.sampler.sample_model(self.x_adj, self.preds[self.mapping[0]], self)

        nuts_kernel = NUTS(sample_model, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=samples, warmup_steps=250)
        mcmc.run(self.x_adj)
        self.samples = mcmc.get_samples()
        return [1]

    def edge_mask(self):
        return self.sampler.edge_mask(self)

    def visualize_subgraph(self, edge_mask=None, threshold=None, **kwargs):
        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, _ = k_hop_subgraph(
            self.node_idx, self.k, self.edge_index, relabel_nodes=True)

        if edge_mask is None:
            edge_mask = self.sampler.edge_mask(self)

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if self.y_true is None:
            self.y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            self.y = self.y_true[subset].to(torch.float) / self.y_true.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=self.y,
                    num_nodes=self.y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'Accent'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G)
        _ = plt.figure(figsize=(10, 10), dpi=300)
        ax = plt.axes()
        ax.axis('off')
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.05),
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=self.y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, ax=ax, **label_kwargs)
        return ax, G
