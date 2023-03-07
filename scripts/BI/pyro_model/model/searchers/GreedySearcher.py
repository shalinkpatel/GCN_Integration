from copy import deepcopy
from .BaseSearcher import BaseSearcher
import pyro
import torch
import pyro.distributions as dist
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
from torch.nn.functional import binary_cross_entropy
import numpy as np


class GreedySearcher(BaseSearcher):
    def __init__(self, name: str, edges: int):
        self.name = name
        self.edges = edges
    
    def search(self, X, y, explainer, **train_hparams):
        ne = explainer.edge_index_adj.shape[1]
        self.ne = ne
        edge_mask = torch.zeros([self.ne])
        G = to_networkx(Data(edge_index=explainer.edge_index_adj, num_nodes=explainer.edge_index_adj.max()))
        nodes = set()
        nodes.add(explainer.mapping.item())
        possible_set = set()
        added_edges = set()

        with torch.no_grad():
            preds = explainer.model(X, explainer.edge_index_adj)[explainer.mapping].reshape(-1).exp().softmax(dim=0).cpu()

        for edge in nx.edges(G, nbunch=list(nodes)):
            possible_set.add((edge[0], edge[1]))
        
        while len(added_edges) < self.edges:
            best = None
            best_ent = 100000
            inc = False

            for consideration in possible_set.difference(added_edges):
                test = deepcopy(added_edges)
                test.add(consideration)
                edge_index_mask = torch.Tensor(np.array([list(map(lambda x: x[0], test)), 
                                                    list(map(lambda x: x[1], test))])).long()
                preds_masked = explainer.model(X, edge_index_mask)[explainer.mapping].reshape(-1).exp()

                curr_ent = binary_cross_entropy(preds, preds_masked).detach().tolist()
                if curr_ent < best_ent:
                    best = consideration
                    best_ent = curr_ent
                    inc = True

            if not inc:
                break

            if best != None:  
                added_edges.add(best)
                start = explainer.edge_index_adj[0, :] == best[0]
                end = explainer.edge_index_adj[1, :] == best[1]
                idx = (start * end).nonzero().item()
                edge_mask[idx] = 1
            
            for edge in nx.edges(G, nbunch=list(nodes)):
                rewrap = (edge[1], edge[0])
                if rewrap not in added_edges:
                    possible_set.add(rewrap)
                if edge not in added_edges:
                    possible_set.add(edge)

            
        return edge_mask

    def run_name(self):
        return self.name
