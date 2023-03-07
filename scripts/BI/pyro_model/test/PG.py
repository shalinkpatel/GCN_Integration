import sys

sys.path.append("../model")

from Experiment import Experiment
from samplers.NFSampler import NFSampler
from samplers.SpikeSlabSampler import SpikeSlabSampler
from samplers.RandomWalkSampler import RandomWalkSampler
from searchers.GNNExplainerSearcher import GNNExplainerSearcher
from searchers.GreedySearcher import GreedySearcher

import torch
from torch_geometric.utils import k_hop_subgraph, from_networkx
import networkx as nx

from julia import Main
from julia import Turing
from julia import Base
from julia import StatsBase
from julia import Optim

from julia import StatsBase
from julia import MCMCChains
import numpy as np

Main.include("rw_model.jl")


experiment = Experiment("syn3-full-verified", "..")
experiment.train_base_model()

N = 529
subset, edge_index_adj, mapping, edge_mask_hard = k_hop_subgraph(
            N, 3, experiment.edge_index, relabel_nodes=True)

total_mapping = {k: i for k, i in enumerate(subset.tolist())}
x_adj = experiment.x[subset]

with torch.no_grad():
    y = experiment.model(x_adj, edge_index_adj).exp()[mapping[0]]

def run_model(edge_list):
    edge_index = torch.Tensor(np.array(edge_list)).long()
    return experiment.model(x_adj, edge_index)[mapping.item()].reshape(-1).exp().detach().numpy()


model = Main.rw_model(edge_index_adj.numpy(), edge_index_adj.max().item(), 0.25, mapping.item(), x_adj, y.numpy(), run_model)
chains = Turing.sample(model, Turing.PG(50), 1000)

arr = Base.Array(chains)
print(arr.mean(axis=0))

