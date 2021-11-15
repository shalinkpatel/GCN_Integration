from operator import pos
import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
from random import choice

from .BaseSampler import BaseSampler


class RandomWalkSampler(BaseSampler):
    def __init__(self, name: str, p: float):
        super().__init__(name)
        self.p = p

    def sample_model(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        sample_dists = torch.full([ne], self.p)

        G = to_networkx(Data(edge_index=explainer.edge_index_adj), num_nodes=explainer.edge_index_adj.max())
        nodes = set()
        nodes.add(0)
        possible_set = set()
        added_edges = set()
        visited = set()

        for edge in nx.edges(G, nbunch=list(nodes)):
            possible_set.add(edge)
        
        while len(possible_set) != 0:
            consideration = choice(list(possible_set))
            possible_set.remove(consideration)
            visited.add(consideration)

            edge = torch.tesnor(list(consideration))
            idx_edge = (explainer.edge_index_adj == edge).nonzero().item()

            include = pyro.sample(f"m_{idx_edge}", dist.Bernoull(sample_dists[idx_edge]).to_event(1))

            if include:
                added_edges.add(consideration)
                nodes.add(consideration[0])
                nodes.add(consideration[1])
            
            for edge in nx.edges(G, nbunch=list(nodes)):
                if edge not in added_edges or edge not in visited:
                    possible_set.add(edge)
            
        Gprime = nx.from_edgelist(list(added_edges))
        edge_index_mask = from_networkx(Gprime).edge_index
        
        mean = explainer.model(X, edge_index_mask)[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        sample_dists = torch.full([ne], self.p)
        m_q = pyro.param("m_q", sample_dists)

        G = to_networkx(Data(edge_index=explainer.edge_index_adj), num_nodes=explainer.edge_index_adj.max())
        nodes = set()
        nodes.add(0)
        possible_set = set()
        added_edges = set()
        visited = set()

        for edge in nx.edges(G, nbunch=list(nodes)):
            possible_set.add(edge)
        
        while len(possible_set) != 0:
            consideration = choice(list(possible_set))
            possible_set.remove(consideration)
            visited.add(consideration)

            edge = torch.tesnor(list(consideration))
            idx_edge = (explainer.edge_index_adj == edge).nonzero().item()

            include = pyro.sample(f"m_{idx_edge}", dist.Bernoull(m_q[idx_edge]).to_event(1))

            if include:
                added_edges.add(consideration)
                nodes.add(consideration[0])
                nodes.add(consideration[1])
            
            for edge in nx.edges(G, nbunch=list(nodes)):
                if edge not in added_edges or edge not in visited:
                    possible_set.add(edge)
            
        Gprime = nx.from_edgelist(list(added_edges))
        edge_index_mask = from_networkx(Gprime).edge_index

        mean = explainer.model(X, edge_index_mask)[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()), obs=y_sample)

    def edge_mask(self, explainer):
        t = pyro.param("m_q")
        return t

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_theta_{self.theta}_alpha-{self.alpha1}-{self.alpha2}_beta-{self.beta1}-{self.beta2}"
