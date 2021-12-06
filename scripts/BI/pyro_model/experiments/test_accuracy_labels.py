import sys
sys.path.append("/users/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/model")

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from torch_geometric.utils import k_hop_subgraph

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm

from Experiment import Experiment
from BayesExplainer import BayesExplainer

experiment = Experiment("syn3-full", "..")
experiment.train_base_model()

label_transform = lambda x, node: x if node < 511 else np.abs(1 - x)

entropies = []
changes = 0

for i in tqdm(range(experiment.x.shape[0])):
    subset, edge_index_adj, mapping, edge_mask_hard = k_hop_subgraph(
            i, 3, experiment.edge_index, relabel_nodes=True)
    x_adj = experiment.x[subset]

    with torch.no_grad():
        preds = experiment.model(x_adj, edge_index_adj)[mapping].reshape(-1).exp().softmax(dim=0)
    
    labs = experiment.labels[edge_mask_hard]
    labs = label_transform(labs, i)
    with torch.no_grad():
        preds_masked = experiment.model(x_adj, edge_index_adj[:, labs == 1])[mapping].reshape(-1).softmax(dim=0)

    entropies.append(binary_cross_entropy(preds, preds_masked).detach().tolist())
    if torch.argmax(preds) != torch.argmax(preds_masked):
        changes += 1

for node_test in [511, 529, 549]:
    subset, edge_index_adj, mapping, edge_mask_hard = k_hop_subgraph(
                node_test, 3, experiment.edge_index, relabel_nodes=True)
    x_adj = experiment.x[subset]
    with torch.no_grad():
            preds = experiment.model(x_adj, edge_index_adj)[mapping].reshape(-1).exp().softmax(dim=0)

    combos = torch.combinations(torch.tensor(list(range(edge_index_adj.shape[1]))), r = 6)

    best_ent = 1000000000
    best_mask = None

    for i in tqdm(range(combos.shape[0])):
        with torch.no_grad():
            preds_masked = experiment.model(x_adj, edge_index_adj[:, combos[i, :]])[mapping].reshape(-1).softmax(dim=0)
        
        curr_ent = binary_cross_entropy(preds, preds_masked).detach().tolist()
        if curr_ent < best_ent:
            best_ent = curr_ent
            best_mask = torch.zeros([edge_index_adj.shape[1]])
            best_mask[combos[i, :]] = 1

    print(f"{node_test} Mask Differential: {(best_mask - experiment.labels[edge_mask_hard]).abs()}")
    print(f"{node_test} Total Difference: {(best_mask - experiment.labels[edge_mask_hard]).abs().sum()}")

    be = BayesExplainer(experiment.model, None, node_test, 3, experiment.x, experiment.data.y, experiment.edge_index)
    be.visualize_subgraph(edge_mask=best_mask)
    plt.savefig(f"tests/{node_test}.png")

sns.lineplot(x=np.arange(len(entropies)), y=np.array(entropies))
plt.savefig("tests/entropies.png")

print(f"Changes: {changes / experiment.x.shape[0]}")