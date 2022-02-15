from re import A
import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
from random import choice, random, choices
import numpy

from .BaseSampler import BaseSampler


class RandomWalkSampler(BaseSampler):
    def __init__(self, name: str, p: float):
        super().__init__(name)
        self.p = p

    def sample_model(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        self.ne = ne
        ps = torch.full([ne], self.p)
        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
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
            idx_edge = (start * end).nonzero().item()

            include = pyro.sample(f"p_{idx_edge}", dist.Bernoulli(ps[idx_edge]))

            if include >= 0.99:
                added_edges.add(consideration)
                added_edges.add((consideration[1], consideration[0]))
                nodes.add(consideration[0])
                nodes.add(consideration[1])
            
            for edge in nx.edges(G, nbunch=list(nodes)):
                rewrap = (edge[1], edge[0])
                if rewrap not in added_edges and rewrap not in visited:
                    possible_set.add(rewrap)
                if edge not in added_edges and edge not in visited:
                    possible_set.add(edge)
            
        Gprime = nx.from_edgelist(list(added_edges))
        edge_index_mask = from_networkx(Gprime).edge_index
        
        mean = explainer.model(X, edge_index_mask)[explainer.mapping].reshape(-1).exp()
        y_sample = choices(list(range(len(y.detach().cpu().tolist()))), y.detach().cpu().tolist())
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean), obs=torch.Tensor(y_sample))

    def sample_guide(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        self.ne = ne
        ps = torch.full([ne], self.p)
        ps = list(map(lambda i: pyro.param(f"p_q_{i}", ps[i], constraint=constraints.unit_interval), range(len(ps))))

        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
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
            idx_edge = (start * end).nonzero().item()
            include = pyro.sample(f"p_{idx_edge}", dist.Bernoulli(ps[idx_edge]))

            if include.item() > 0.99:
                added_edges.add(consideration)
                added_edges.add((consideration[1], consideration[0]))
                nodes.add(consideration[0])
                nodes.add(consideration[1])
            
            for edge in nx.edges(G, nbunch=list(nodes)):
                rewrap = (edge[1], edge[0])
                if rewrap not in added_edges and rewrap not in visited:
                    possible_set.add(rewrap)
                if edge not in added_edges and edge not in visited:
                    possible_set.add(edge)
            
        Gprime = nx.from_edgelist(list(added_edges))
        edge_index_mask = from_networkx(Gprime).edge_index

        mean = explainer.model(X, edge_index_mask)[explainer.mapping].reshape(-1).exp()
        y_sample = choices(list(range(len(y.detach().cpu().tolist()))), y.detach().cpu().tolist())

    def edge_mask(self, explainer):
        edge_mask = torch.zeros([self.ne])
        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
        for _ in range(1000):
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
                idx_edge = (start * end).nonzero().item()

                include = random() < pyro.param(f"p_q_{idx_edge}").item()

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
                    if edge not in added_edges and edge not in visited:
                        possible_set.add(edge)
        return edge_mask / 1000

    def ret_probs(self, explainer):
        return list(map(lambda i: f"({explainer.total_mapping[explainer.edge_index_adj[0, i].item()]}, "
                                  f"{explainer.total_mapping[explainer.edge_index_adj[1, i].item()]}): "
                                  f"{pyro.param(f'p_q_{i}')}", range(self.ne)))

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}p-{self.p}"
