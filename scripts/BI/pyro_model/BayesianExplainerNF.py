import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import pyro
import pyro.distributions as dist
from torch_geometric.utils import k_hop_subgraph, to_networkx
import torch.distributions.constraints as constraints
from torch_geometric.data import Data
import networkx as nx
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine
from copy import copy
from math import sqrt
import pyro.distributions.transforms as T

class BayesianExplainer():
    def __init__(self, model, node_idx: int, k: int, x, edge_index, sharp: float = 0.01, splines: int = 6):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.base_dist = dist.Normal(torch.zeros(self.N).to(device), torch.ones(self.N).to(device))
        self.splines = []
        for i in range(splines):
            self.splines.append(T.spline(self.N).to(device))
        self.flow_dist = dist.TransformedDistribution(self.base_dist,self.splines)
    
    def sample_model(self, X, y):
        m = pyro.sample("m", dist.Bernoulli(self.flow_dist.rsample(torch.Size([100,])).sigmoid().mean(dim=0)).to_event(1))
        mean = self.model(X, self.edge_index_adj[:, m == 1])[self.mapping].reshape(-1)
        sigma = (self.sharp * torch.eye(mean.size(0))).to(self.device)
        y_sample = pyro.sample("y", dist.MultivariateNormal(mean, sigma), obs = y)
    
    def sample_guide(self, X, y):
        modules = []
        for (i, spline) in enumerate(self.splines):
            modules.append(pyro.module("spline%i" % i, spline))
        
    def ma(self, l, window):
        cumsum, moving_aves = [0], []

        for i, x in enumerate(l, 1):
            cumsum.append(cumsum[i-1] + x)
            if i>=window:
                moving_ave = (cumsum[i] - cumsum[i-window])/window
                #can do stuff with moving_ave here
                moving_aves.append(moving_ave)
        return moving_aves
    
    def L(self, p):
        return self.flow_dist.rsample(torch.Size([100,])).sigmoid().pow(p).mean()

    def train(self, epochs: int = 3000, lr: float = 0.005, lambd: float = 0.1, window: int = 1000, p = 1.25, log: bool = True):
        pyro.get_param_store().clear()
        params = []
        for spline in self.splines:
            params += spline.parameters()
        self.p = torch.nn.parameter.Parameter(torch.tensor(p).float())
        params += [self.p]
        optimizer = torch.optim.Adam(params, lr)
        loss_fn = lambda model, guide, X, y: pyro.infer.Trace_ELBO().differentiable_loss(model, guide, X, y) * self.sharp + lambd * self.L(self.p)

        n_steps = epochs
        # do gradient steps
        if log:
            pbar = tqdm(range(n_steps))
        else:
            pbar = range(n_steps)
        elbos = []
        for step in pbar:
            loss = loss_fn(self.sample_model, self.sample_guide, self.x_adj, self.preds[self.mapping[0]])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            elbos.append(loss.cpu().detach().item())
            avgs = self.ma(elbos, window)
            if step >= window:
                disp = avgs[-1]
            else:
                disp = loss.cpu().detach().item()
            if log:
                pbar.set_description("Loss -> %.4f" % disp)
        return avgs
    
    def edge_mask(self):
        return self.flow_dist.rsample(torch.Size([10000,])).sigmoid().mean(dim=0)

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           k = 2, threshold=None, **kwargs):
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """

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