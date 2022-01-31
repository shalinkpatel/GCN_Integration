from copy import copy
from os.path import exists
from shutil import rmtree

from math import sqrt

import matplotlib.pyplot as plt
import networkx as nx
import pyro
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from tqdm.autonotebook import tqdm

from searchers.BaseSearcher import BaseSearcher

class DeterministicExplainer:
    def __init__(self, model: torch.nn.Module, searcher: BaseSearcher, node_idx: int, k: int,
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
        self.x_adj = self.x[self.subset]
        self.device = device

        with torch.no_grad():
            self.preds = model(self.x, self.edge_index_adj).exp()

        self.N = self.edge_index_adj.size(1)

        self.searcher = searcher

    def edge_mask(self, base: str = ".", **train_hparams):
        edge_mask = self.searcher.search(self.x_adj, self.preds[self.mapping[0]], self, **train_hparams)
        self.edge_mask = edge_mask
        name = self.sampler.run_name()
        path = f"{base}/runs/individual/{name}"
        if exists(path):
            rmtree(path)
        writer = SummaryWriter(path)
        ax, _ = self.visualize_subgraph()
        writer.add_figure("Importance Graph", ax.get_figure(), 0)
        writer.add_histogram("Importances", edge_mask, 0)
        return edge_mask

    def visualize_subgraph(self, edge_mask=None, threshold=None, **kwargs):
        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, _ = k_hop_subgraph(
            self.node_idx, self.k, self.edge_index, relabel_nodes=True)

        if edge_mask is None:
            edge_mask = self.edge_mask

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
        nx.draw_networkx_nodes(G, pos, node_color=self.y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G
