from model.samplers.NFGradSampler import NFGradSampler
from model.BayesExplainer import BayesExplainer

import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import chain, repeat
from random import shuffle
import sys
import time

from torch_geometric.nn import SAGEConv, GraphConv
from torch_geometric.nn import global_add_pool
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.metric.basic import groundtruth_metrics
from torch_geometric.utils import k_hop_subgraph
from torch.multiprocessing import Pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from os.path import exists
from typing import Union, Tuple
from copy import deepcopy

from random import random

procs = 8

# Definitions
class Model(torch.nn.Module):
    def __init__(self, y, N, x):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(1, x)
        self.conv2 = SAGEConv(x, 2*x)
        self.conv3 = SAGEConv(2*x, x)
        self.fc1 = torch.nn.Linear(x, y.max() + 1)
        self.N = N

    def forward(self, x, edge_index, batch = None):
        if batch is None:
            batch = torch.ones(self.N).to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        return self.fc1(x)


def train_model(model, X, y, edge_index, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    best_acc = 0
    print('=' * 20 + ' Started Training ' + '=' * 20)
    pbar = range(3000)
    best_weights = None
    for epoch in pbar:
        # Training step
        model.train()
        loss_ep = 0
        correct = 0
        avg_max = 0
        rand_check = round(random() * y.shape[0])
        idxs = list(range(y.shape[0]))
        shuffle(idxs)
        data_list = [Data(x = X[:, n:n+1], y=y[n], edge_index=edge_index) for n in idxs]
        loader = DataLoader(data_list, batch_size=16)
        for grp in loader:
            optimizer.zero_grad()
            logits = model(grp.x, grp.edge_index, grp.batch)
            loss = F.cross_entropy(logits, grp.y)
            probs = logits.softmax(dim=1)
            avg_max += probs.detach().amax(dim=1).sum().item()
            loss_ep += loss.detach().item()
            loss.backward()
            optimizer.step()
            correct += (torch.argmax(probs, dim=1) == grp.y).sum().float().item()
        avg_max /= y.shape[0]

        # Testing step
        best_acc = correct / y.shape[0] if (correct / y.shape[0]) > best_acc else best_acc
        if best_acc == correct / y.shape[0]:
            model.to(torch.device('cpu'))
            best_weights = deepcopy(model.state_dict())
            model.to(device)
        print(f"Epoch {epoch} | Best Acc = {best_acc} | Loss = {loss_ep / len(idxs)} | Avg Max = {avg_max}")
    print('=' * 20 + ' Ended Training ' + '=' * 20)  
    correct = 0
    for n in range(y.shape[0]):
        probs = model(X[:, n:n + 1], edge_index)
        correct += (torch.argmax(probs) == y[n].item()).float().item()
    acc = correct / y.shape[0]
    print(acc)
    return best_weights


def save_masks(name: str, grn: torch.Tensor, exp: torch.Tensor, ei: torch.Tensor):
    with open(f'experiments/masks/{name}_{groups}.csv', 'w') as f:
        f.write('s,d,grn,exp\n')
        for s, d, g, e in zip(ei[0, :], ei[1, :], grn, exp):
            f.write(f'{s.item()},{d.item()},{g.item()},{e.item()}\n')

def train_nfg_model(device: torch.device, model: Model, node: int,
        X: torch.Tensor, y: torch.Tensor,
        G: torch.Tensor, nfg_hparams: dict) -> Tuple[int, Union[torch.Tensor, None]]:
    nfg_sampler = NFGradSampler(device=device, **nfg_hparams)
    explainer = BayesExplainer(model, nfg_sampler, node, 3, X, y, G, True, device)
    if explainer.edge_index_adj.shape[1] == 0:
        return (node, None)
    explainer.train(epochs=1250, lr=0.001, window=500, log=False)
    res = explainer.edge_mask()
    return (node, res)

if __name__ == '__main__':
    # Loading Data
    groups = sys.argv[1]
    device = torch.device('cuda')

    X = torch.load(
        f"model/sergio/final_data/100gene-{groups}groups"
        f"-1sparsity.pt").float()
    y = torch.load(
        f"model/sergio/final_data/100gene-{groups}groups"
        f"-labels.pt")
    G = torch.load(
        f"model/sergio/final_data/100gene-{groups}groups"
        f"-1sparsity-compgraph.pt")
    grn = torch.load(
        f"model/sergio/final_data/100gene-{groups}groups"
        f"-gt-grn.pt")

    X = X.to(device)
    y = y.to(device)
    G = G.to(device)
    grn = grn.to(device)
    grn_s = set(
        [(s.cpu().item(), d.cpu().item()) for s, d in zip(chain(grn[0, :], grn[1, :]), chain(grn[1, :], grn[0, :]))])
    gt_grn = torch.tensor([1 if (s.cpu().item(), d.cpu().item()) in grn_s else 0 for s, d in zip(G[0, :], G[1, :])]).to(
        device)

    model = Model(y, 100, 128)
    model.to(device)


    # Model Training
    if exists(f"experiments/models/graph_class_{groups}groups.pt"):
        print('=' * 20 + " Loading Previous Model " + '=' * 20)
        model.load_state_dict(torch.load(f"experiments/models/graph_class_{groups}groups.pt", map_location=torch.device('cpu')))
        model.to(device)
    else:
        print('=' * 20 + " Training Model " + '=' * 20)
        sd = train_model(model, X, y, G, device)
        torch.save(sd, f"experiments/models/graph_class_{groups}groups.pt")
    exit()

    m_names = ["accuracy", "recall", "precision", "f1_score", "auroc"]


    # NFG Explainer
    print('=' * 20 + " NFG Explainer " + '=' * 20)
    metrics_nf_grad = [0, 0, 0, 0, 0]
    nfg_hparams = {
        "name": "normalizing_flows_grad",
        "splines": 12,
        "sigmoid": True,
        "lambd": 5.0,
        "p": 1.2,
    }
    samples = list(range(y.shape[0]))
    shuffle(samples)
    n_samples = 0
    graph = 0
    final_nfgexp_explanation = torch.zeros_like(gt_grn).float()
    avg_nfgexp_explanation = torch.zeros_like(gt_grn).float()
    avg_nfgexp_touched = torch.zeros_like(gt_grn).float()
    mp_pool = Pool(processes=procs)
    for x in samples[:int(0.10 * len(samples))]:
        graph += 1
        nodes = list(range(X.shape[0]))
        shuffle(nodes)
        start = time.time()
        print(f"Starting training run for graph {graph}")
        results = mp_pool.starmap(train_nfg_model, zip(repeat(device), repeat(model),
            nodes[:procs], repeat(X[:,x:x+1]), repeat(y), repeat(G), repeat(nfg_hparams)))
        print(f"Time for graph {graph}: {time.time() - start}")
        for n, res in results:
            if res is None:
                continue
            _, _, _, edge_mask_hard = k_hop_subgraph(n, 3, G)
            final_nfgexp_explanation[edge_mask_hard] = torch.max(final_nfgexp_explanation[edge_mask_hard], res)
            avg_nfgexp_explanation[edge_mask_hard] += res
            avg_nfgexp_touched[edge_mask_hard] += 1
            res = groundtruth_metrics(res, gt_grn[edge_mask_hard])
            metrics_nf_grad = [m + r for m, r in zip(metrics_nf_grad, res)]
            n_samples += 1
    mp_pool.close()
    mp_pool.join()
    metrics_nf_grad = [m / n_samples for m in metrics_nf_grad]
    avg_nfgexp_explanation /= avg_nfgexp_touched
    print('=' * 20 + " NFG Results " + '=' * 20)
    print({n: v for n, v in zip(m_names, metrics_nf_grad)})
    print({n: v for n, v in zip(m_names, groundtruth_metrics(final_nfgexp_explanation, gt_grn))})
    print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_nfgexp_explanation, gt_grn))})
    save_masks("nfgexp_max_mask", gt_grn, final_nfgexp_explanation, G)
    save_masks("nfgexp_avg_mask", gt_grn, avg_nfgexp_explanation, G)


    # GNN Explainer
    print('=' * 20 + " GNN Explainer " + '=' * 20)
    metrics_gnn_exp = [0, 0, 0, 0, 0]
    gnn_explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=1000),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    )
    final_gnnexp_explanation = torch.zeros_like(gt_grn).float()
    avg_gnnexp_explanation = torch.zeros_like(gt_grn).float()
    for x in range(y.shape[0]):
        gnn_exp_explanation = gnn_explainer(X[:, x:x + 1], G)
        final_gnnexp_explanation = torch.max(final_gnnexp_explanation, gnn_exp_explanation.edge_mask)
        avg_gnnexp_explanation += gnn_exp_explanation.edge_mask
        res = groundtruth_metrics(gnn_exp_explanation.edge_mask, gt_grn)
        metrics_gnn_exp = [m + r for m, r in zip(metrics_gnn_exp, res)]
    metrics_gnn_exp = [m / y.shape[0] for m in metrics_gnn_exp]
    avg_gnnexp_explanation /= y.shape[0]
    print('=' * 20 + " GNNExp Results " + '=' * 20)
    print({n: v for n, v in zip(m_names, metrics_gnn_exp)})
    print({n: v for n, v in zip(m_names, groundtruth_metrics(final_gnnexp_explanation, gt_grn))})
    print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_gnnexp_explanation, gt_grn))})
    save_masks("gnnexp_max_mask", gt_grn, final_gnnexp_explanation, G)
    save_masks("gnnexp_avg_mask", gt_grn, avg_gnnexp_explanation, G)


    # PG Explainers
    print('=' * 20 + " PG Explainer " + '=' * 20)
    metrics_pg_exp = [0, 0, 0, 0, 0]
    pg_explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=100 * y.shape[0], lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        )
    )
    for epoch in range(100):
        for x in range(y.shape[0]):
            pg_explainer.algorithm.train(epoch * y.shape[0] + x, model, X[:, x:x + 1], G, target=y[x])
    final_pgexp_explanation = torch.zeros_like(gt_grn).float()
    avg_pgexp_explanation = torch.zeros_like(gt_grn).float()
    for x in range(y.shape[0]):
        pg_exp_explanation = pg_explainer(X[:, x:x + 1], G, target=y[x])
        final_pgexp_explanation = torch.max(final_pgexp_explanation, pg_exp_explanation.edge_mask)
        avg_pgexp_explanation += pg_exp_explanation.edge_mask
        res = groundtruth_metrics(pg_exp_explanation.edge_mask, gt_grn)
        metrics_gnn_exp = [m + r for m, r in zip(metrics_gnn_exp, res)]
    metrics_gnn_exp = [m / y.shape[0] for m in metrics_gnn_exp]
    avg_pgexp_explanation /= y.shape[0]
    print('=' * 20 + " PGExp Results " + '=' * 20)
    print({n: v for n, v in zip(m_names, metrics_gnn_exp)})
    print({n: v for n, v in zip(m_names, groundtruth_metrics(final_pgexp_explanation, gt_grn))})
    print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_pgexp_explanation, gt_grn))})
    save_masks("pgexp_max_mask", gt_grn, final_pgexp_explanation, G)
    save_masks("pgexp_avg_mask", gt_grn, avg_pgexp_explanation, G)
