from copy import copy
from math import sqrt

import matplotlib.pyplot as plt
import networkx as nx
import pyro
import torch
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from tqdm import tqdm

from .samplers.BaseSampler import BaseSampler


class BayesExplainer:
    def __init__(self, model: torch.nn.Module, sampler: BaseSampler, node_idx: int, k: int,
            x: torch.Tensor, edge_index: torch.Tensor, sharp: float):
        device = torch.device('cpu')
        self.model = model.to(device)
        self.x = x.to(device)
        self.edge_index = edge_index.to(device)
        self.node_idx = node_idx
        self.k = k
        self.sharp = sharp
        self.subset, self.edge_index_adj, self.mapping, self.edge_mask_hard = k_hop_subgraph(
            self.node_idx, k, self.edge_index, relabel_nodes=True)
        self.x_adj = self.x[self.subset]
        self.device = device

        with torch.no_grad():
            self.preds = model(self.x, self.edge_index_adj)
        
        self.N = self.edge_index_adj.size(1)

        self.sampler = sampler

    def ma(self, l, window):
        cumsum, moving_aves = [0], []

        for i, x in enumerate(l, 1):
            cumsum.append(cumsum[i-1] + x)
            if i>=window:
                moving_ave = (cumsum[i] - cumsum[i-window])/window
                moving_aves.append(moving_ave)
        return moving_aves

    def train(self, epochs: int = 3000, lr: float = 0.005, window: int = 1000, log: bool = True):
        pyro.get_param_store().clear()
        adam_params = {"lr": lr, "betas": (0.95, 0.999)}
        optimizer = Adam(adam_params)
        # setup the inference algorithm
        svi = SVI(self.sampler.sample_model, self.sampler.sample_guide, optimizer, loss=Trace_ELBO())

        n_steps = epochs
        # do gradient steps
        if log:
            pbar = tqdm(range(n_steps))
        else:
            pbar = range(n_steps)
        elbos = []
        for step in pbar:
            elbo = svi.step(self.x_adj, self.preds[self.mapping[0]], self)
            elbos.append(elbo)
            avgs = self.ma(elbos, window)
            if step >= window:
                disp = avgs[-1]
            else:
                disp = elbo
            if log:
                pbar.set_description("Loss -> %.4f" % disp)
        return avgs
    
    def edge_mask(self):
        return self.sampler.edge_mask(self)

    @staticmethod
    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           k = 2, threshold=None, **kwargs):
        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, _ = k_hop_subgraph(
            node_idx, k, edge_index, relabel_nodes=True)

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'Accent'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G)
        ax = plt.gca()
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
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G