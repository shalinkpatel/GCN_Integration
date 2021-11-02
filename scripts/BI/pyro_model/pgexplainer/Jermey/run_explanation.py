import sys
import argparse

sys.path.append("..")
from PGExplainer import PGExplainer

from models import GCN_classification
import torch
import pickle
import numpy as np
from scipy.sparse import load_npz
import torch_geometric
from torch_geometric.utils import k_hop_subgraph

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int)
args = parser.parse_args()

def return_subgraph(n: int, k: int, X: torch.Tensor, edge_index: torch.Tensor):
    subset, edge_index_adj, mapping, edge_mask_hard = k_hop_subgraph(n, k, edge_index, relabel_nodes=True)
    x_adj = X[subset]
    return x_adj, edge_index_adj

node_exp = args.n

model = GCN_classification(6, 2, [6, 256, 256], 3, [256, 256, 256, 2], 2)
model.embedding_size = 256
model.load_state_dict(torch.load('data/E116/model_2021-06-26-at-05-46-03.pt', map_location=torch.device('cpu')))
model.eval()

data = np.load('data/E116/np_hmods_norm_chip_10000bp.npy')
mat = load_npz('data/E116/hic_sparse.npz')
extract = torch_geometric.utils.from_scipy_sparse_matrix(mat)

edge_index = extract[0]

nodes = torch.tensor(data[:, 0].astype(np.long))
X = torch.tensor(data[:, 1:]).float()

X_adj, edge_index_adj = return_subgraph(node_exp, 3, X, edge_index)

pg_explainer = PGExplainer(model, edge_index_adj, X_adj, "node")
pg_explainer.prepare()
print(pg_explainer.explain(node_exp))

