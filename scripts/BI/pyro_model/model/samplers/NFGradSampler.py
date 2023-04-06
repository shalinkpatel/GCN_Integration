import pyro
import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T

import networkx as nx
from random import random, choice

from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.data import Data

from .BaseSampler import BaseSampler


class NFGradSampler(BaseSampler):
    def __init__(self, name, splines: int, sigmoid: bool, lambd: float, p: float, device: torch.device):
        super().__init__(name)
        self.sigmoid = sigmoid
        self.splines_n = splines
        self.device = device

        self.init = False

        self.lambd = lambd
        self.p = p

    def _init_node(self, N):
        self.base_dist = dist.Normal(torch.zeros(N).to(self.device), torch.ones(N).to(self.device))
        self.splines = []
        for _ in range(self.splines_n):
            self.splines.append(T.spline(N).to(self.device))
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.splines)

    def sample_model(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        self.ne = ne
        if not self.init:
            self._init_node(explainer.edge_index_adj.shape[1])
        m_sub = self.flow_dist.rsample(torch.Size([250, ]))
        if self.sigmoid:
            m_sub = m_sub.sigmoid().clamp(0, 1).mean(dim=0)
        else:
            m_sub = m_sub.clamp(0, 1).mean(dim=0)
        if explainer.mixed_mode:
            new_mask = torch.ones_like(explainer.edge_mask_hard).float().to(self.device)
            new_mask[explainer.edge_mask_hard] = m_sub
            m_sub = new_mask
        set_masks(explainer.model, m_sub, explainer.final_ei, False)
        if explainer.mixed_mode:
            mean = explainer.model(X, explainer.final_ei).reshape(-1).exp()
        else:
            mean = explainer.model(X, explainer.final_ei)[explainer.mapping].reshape(-1).exp()
        clear_masks(explainer.model)
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y))
        return pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        if not self.init:
            self._init_node(explainer.edge_index_adj.shape[1])
        modules = []
        for (i, spline) in enumerate(self.splines):
            modules.append(pyro.module(f"spline{i}", spline))
        m_sub = self.flow_dist.rsample(torch.Size([250, ]))
        if self.sigmoid:
            m_sub = m_sub.sigmoid().clamp(0, 1).mean(dim=0)
        else:
            m_sub = m_sub.clamp(0, 1).mean(dim=0)
        if explainer.mixed_mode:
            new_mask = torch.ones_like(explainer.edge_mask_hard).float().to(self.device)
            new_mask[explainer.edge_mask_hard] = m_sub
            m_sub = new_mask
        set_masks(explainer.model, m_sub, explainer.final_ei, False)
        if explainer.mixed_mode:
            mean = explainer.model(X, explainer.final_ei).reshape(-1).exp()
        else:
            mean = explainer.model(X, explainer.final_ei)[explainer.mapping].reshape(-1).exp()
        clear_masks(explainer.model)
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean))

    def edge_mask(self, explainer):
        sample = self.flow_dist.rsample(torch.Size([250, ]))
        sample = sample.sigmoid() if self.sigmoid else sample
        sample = sample.clamp(0, 1)
        post = sample.mean(dim=0)

        edge_mask = torch.zeros([self.ne])
        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
        for i in range(250):
            nodes = set()
            nodes.add(explainer.mapping.item())
            possible_set = set()
            added_edges = set()
            visited = set()

            for edge in nx.edges(G, nbunch=list(nodes)):
                possible_set.add((edge[1], edge[0]))
                possible_set.add((edge[0], edge[1]))

            while len(possible_set) != 0:
                consideration = choice(list(possible_set))
                possible_set.remove(consideration)
                visited.add(consideration)

                start = explainer.edge_index_adj[0, :] == consideration[0]
                end = explainer.edge_index_adj[1, :] == consideration[1]

                idx_edge = (start * end).nonzero()
                if idx_edge.numel() == 0:
                    include = False
                else:
                    idx_edge = idx_edge.item()
                    include = random() < post[idx_edge]

                if include:
                    added_edges.add(consideration)
                    added_edges.add((consideration[1], consideration[0]))
                    edge_mask[idx_edge] += 1
                    nodes.add(consideration[0])
                    nodes.add(consideration[1])

                for edge in nx.edges(G, nbunch=list(nodes)):
                    rewrap = (edge[1], edge[0])
                    if rewrap not in added_edges and rewrap not in visited:
                        possible_set.add(rewrap)
                    if edge not in added_edges and rewrap not in visited:
                        possible_set.add(edge)
        masking = edge_mask / 1000
        return masking
        # return post.detach()

    def L(self, p):
        sample = self.flow_dist.rsample(torch.Size([250, ]))
        s2 = sample.sigmoid() if self.sigmoid else sample.clamp(0, 1)
        sample = s2.pow(p)
        sample = sample / sample.max()
        return sample.mean()

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args) + self.lambd * self.L(self.p)

    def run_name(self):
        return f"{self.name}_splines-{self.splines_n}_sig-{self.sigmoid}_lambd-{self.lambd}_p-{self.p}"
