from operator import pos
import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
from random import choice, sample

from .BaseSampler import BaseSampler


class RandomWalkSampler(BaseSampler):
    def __init__(self, name: str, p: float):
        super().__init__(name)
        self.p = p

    def sample_model(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        self.ne = ne
        sample_dists = torch.full([ne], self.p)
        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
        nodes = set()
        nodes.add(0)
        possible_set = set()
        added_edges = set()
        visited = set()

        for edge in nx.edges(G, nbunch=list(nodes)):
            possible_set.add((edge[1], edge[0]))
        
        while len(possible_set) != 0:
            consideration = choice(list(possible_set))
            possible_set.remove(consideration)
            visited.add(consideration)

            start = explainer.edge_index_adj[0, :] == consideration[0]
            end = explainer.edge_index_adj[1, :] == consideration[1]
            idx_edge = (start * end).nonzero().item()

            include = pyro.sample(f"m_{idx_edge}", dist.Bernoulli(sample_dists[idx_edge]))

            if include >= 0.99:
                added_edges.add(consideration)
                nodes.add(consideration[0])
                nodes.add(consideration[1])
            
            for edge in nx.edges(G, nbunch=list(nodes)):
                rewrap = (edge[1], edge[0])
                if rewrap not in added_edges and rewrap not in visited:
                    possible_set.add(rewrap)
            
        Gprime = nx.from_edgelist(list(added_edges))
        edge_index_mask = from_networkx(Gprime).edge_index
        
        mean = explainer.model(X, edge_index_mask)[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        self.ne = ne
        sample_dists = torch.full([ne], self.p)
        sample_dists = list(map(lambda i: pyro.param(f"m_q_{i}", sample_dists[i], constraint=constraints.unit_interval), range(len(sample_dists))))

        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
        nodes = set()
        nodes.add(0)
        possible_set = set()
        added_edges = set()
        visited = set()

        for edge in nx.edges(G, nbunch=list(nodes)):
            possible_set.add((edge[1], edge[0]))
        
        while len(possible_set) != 0:
            consideration = choice(list(possible_set))
            possible_set.remove(consideration)
            visited.add(consideration)

            start = explainer.edge_index_adj[0, :] == consideration[0]
            end = explainer.edge_index_adj[1, :] == consideration[1]
            idx_edge = (start * end).nonzero().item()

            include = pyro.sample(f"m_{idx_edge}", dist.Bernoulli(sample_dists[idx_edge]))

            if include.item() > 0.99:
                added_edges.add(consideration)
                nodes.add(consideration[0])
                nodes.add(consideration[1])
            
            for edge in nx.edges(G, nbunch=list(nodes)):
                rewrap = (edge[1], edge[0])
                if rewrap not in added_edges and rewrap not in visited:
                    possible_set.add(rewrap)
            
        Gprime = nx.from_edgelist(list(added_edges))
        edge_index_mask = from_networkx(Gprime).edge_index

        mean = explainer.model(X, edge_index_mask)[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()), obs=y_sample)

    def edge_mask(self, explainer):
        edge_mask = torch.zeros([self.ne])
        sample_dists = list(map(lambda i: pyro.param(f"m_q_{i}"), range(self.ne)))
        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
        for _ in range(1000):
            nodes = set()
            nodes.add(0)
            possible_set = set()
            added_edges = set()
            visited = set()

            for edge in nx.edges(G, nbunch=list(nodes)):
                possible_set.add((edge[1], edge[0]))
            
            while len(possible_set) != 0:
                consideration = choice(list(possible_set))
                possible_set.remove(consideration)
                visited.add(consideration)

                start = explainer.edge_index_adj[0, :] == consideration[0]
                end = explainer.edge_index_adj[1, :] == consideration[1]
                idx_edge = (start * end).nonzero().item()

                include = pyro.sample(f"m_{idx_edge}", dist.Bernoulli(sample_dists[idx_edge]))

                if include.item() > 0.99:
                    added_edges.add(consideration)
                    edge_mask[idx_edge] += 1
                    nodes.add(consideration[0])
                    nodes.add(consideration[1])
                
                for edge in nx.edges(G, nbunch=list(nodes)):
                    rewrap = (edge[1], edge[0])
                    if rewrap not in added_edges and rewrap not in visited:
                        possible_set.add(rewrap)
        return edge_mask / 1000

    def ret_probs(self, explainer):
        return list(map(lambda i: pyro.param(f"m_q_{i}"), range(self.ne)))

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_p_{self.p}"
