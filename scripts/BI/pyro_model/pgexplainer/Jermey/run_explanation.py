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
import networkx as nx
import matplotlib.pyplot as plt
import math
from pathlib import Path

from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int)
args = parser.parse_args()

def return_subgraph(n: int, k: int, X: torch.Tensor, edge_index: torch.Tensor):
    subset, edge_index_adj, mapping, edge_mask_hard = k_hop_subgraph(n, k, edge_index, relabel_nodes=True)
    x_adj = X[subset]
    return x_adj, edge_index_adj, mapping

def plot(node_idx, edge_index, edge_mask, y, threshold=None, args={}, **kwargs):
        data = torch_geometric.data.Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = torch_geometric.utils.to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

        node_kwargs = {}
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'Accent'

        label_kwargs = {}
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
                    shrinkA=math.sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=math.sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        save_path = f'./qualitative/e_{args.explainer}/m_{args.model}/d_{args.dataset}/'

        # Generate folders if they do not exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Save figure
        plt.savefig(f'{save_path}{node_idx}.png')
        plt.clf()

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

X_adj, edge_index_adj, node_exp = return_subgraph(node_exp, 2, X, edge_index)

pg_explainer = PGExplainer(model, edge_index_adj, X_adj, "node", epochs=3000, lr=0.0003, reg_coefs=(0.1, 0.5))
pg_explainer.prepare()
graph, weights = pg_explainer.explain(node_exp)

print(weights)
plot(node_exp.numpy().tolist()[0], graph, weights, torch.ones(X_adj.shape[0]), args=SimpleNamespace(**{'explainer': 'pg_explainer', 'model': 'class', 'dataset': 'E116'}))