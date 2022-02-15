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


class ConstrainedRandomWalkSampler(BaseSampler):
    def __init__(self, name: str, alpha: float, beta: float):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta

    def sample_model(self, X, y, explainer):
        ne = explainer.edge_index_adj.shape[1]
        self.ne = ne
        sample_alphas = torch.full([ne], self.alpha)
        sample_betas = torch.full([ne], self.beta)
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

            chance = pyro.sample(f"p_{idx_edge}", dist.Beta(sample_alphas[idx_edge], sample_betas[idx_edge]))
            include = pyro.sample(f"m_{idx_edge}", dist.Bernoulli(chance))

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
        sample_alphas = torch.full([ne], self.alpha)
        sample_betas = torch.full([ne], self.beta)
        sample_alphas = list(map(lambda i: pyro.param(f"a_q_{i}", sample_alphas[i], constraint=constraints.positive), range(len(sample_alphas))))
        sample_betas = list(map(lambda i: pyro.param(f"b_q_{i}", sample_betas[i], constraint=constraints.positive), range(len(sample_betas))))

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

            chance = pyro.sample(f"p_{idx_edge}", dist.Beta(sample_alphas[idx_edge], sample_betas[idx_edge]))
            include = pyro.sample(f"m_{idx_edge}", dist.Bernoulli(chance))

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
        m = torch.zeros([self.ne])
        for i in range(self.ne):
            m[i] = dist.Beta(pyro.param(f"a_q_{i}"), pyro.param(f"b_q_{i}")).mean
        return m

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_alpha_{self.alpha}_beta_{self.beta}"
