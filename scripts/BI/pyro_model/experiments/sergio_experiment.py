from model.samplers.NFGradSampler import NFGradSampler
from model.BayesExplainer import BayesExplainer

import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import chain
from random import shuffle
import sys

from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.metric.basic import groundtruth_metrics
from torch_geometric.utils import k_hop_subgraph

from os.path import exists

groups = sys.argv[1]

# Definitions
class Model(torch.nn.Module):
    def __init__(self, y, N, x=64):
        super(Model, self).__init__()
        self.conv1 = GCNConv(1, 10)
        self.conv2 = GCNConv(10, x)
        self.conv3 = GCNConv(x, x)
        self.fc = torch.nn.Linear(x * N, max(y).tolist() + 1)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        return self.fc(x.flatten()).softmax(dim=0)


def train_model(model, X, y, edge_index, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0
    print('=' * 20 + ' Started Training ' + '=' * 20)
    pbar = range(3000)
    best_weights = None
    for epoch in pbar:
        # Training step
        model.train()
        loss_ep = 0
        correct = 0
        for n in range(y.shape[0]):
            optimizer.zero_grad()
            log_logits = model(X[:, n:n + 1], edge_index)
            loss = F.cross_entropy(log_logits, y[n])
            loss_ep += loss
            loss.backward()
            optimizer.step()
            correct += (torch.argmax(log_logits) == y[n].item()).float().item()

        # Testing step
        model.eval()
        best_acc = correct / y.shape[0] if (correct / y.shape[0]) > best_acc else best_acc
        if best_acc == correct / y.shape[0]:
            model.to(torch.device('cpu'))
            best_weights = model.state_dict()
            model.to(device)
        if (epoch + 1 % 100) == 0:
            print(f"Epoch {epoch} | Best Acc = {best_acc} | Loss = {loss_ep}")
    print('=' * 20 + ' Ended Training ' + '=' * 20)
    return best_weights


def save_masks(name: str, grn: torch.Tensor, exp: torch.Tensor, ei: torch.Tensor):
    with open(f'{name}_{groups}.csv', 'w') as f:
        f.write('s,d,grn,exp\n')
        for s, d, g, e in zip(ei[0, :], ei[1, :], grn, exp):
            f.write(f'{s.item()},{d.item()},{g.item()},{e.item()}\n')


# Loading Data
device = torch.device('cpu')

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

model = Model(y, 100)
model.to(device)


# Model Training
if exists(f"experiments/models/graph_class_{groups}groups.pt"):
    print('=' * 20 + "Loading Previous Model" + '=' * 20)
    model.load_state_dict(torch.load(f"experiments/models/graph_class_{groups}groups.pt", map_location=torch.device('cpu')))
    model.to(device)
else:
    print('=' * 20 + "Training Model" + '=' * 20)
    sd = train_model(model, X, y, G, device)
    torch.save(sd, f"experiments/models/graph_class_{groups}groups.pt")

m_names = ["accuracy", "recall", "precision", "f1_score", "auroc"]


# NFG Explainer
print('=' * 20 + "NFG Explainer" + '=' * 20)
metrics_nf_grad = [0, 0, 0, 0, 0]
nfg_hparams = {
    "name": "normalizing_flows_grad",
    "splines": 6,
    "sigmoid": True,
    "lambd": 5.0,
    "p": 1.5,
}
samples = list(range(y.shape[0]))
shuffle(samples)
for x in tqdm(samples[:int(0.05 * len(samples))]):
    nodes = list(range(X.shape[0]))
    shuffle(nodes)
    for n in tqdm(nodes[:int(0.1 * len(nodes))]):
        nfg_sampler = NFGradSampler(device=device, **nfg_hparams)
        explainer = BayesExplainer(model, nfg_sampler, n, 3, X[:, x:x + 1], y, G, True, device)
        if explainer.edge_index_adj.shape[1] == 0:
            continue
        explainer.train(epochs=250, lr=0.001, window=500, log=False)
        res = explainer.edge_mask()
        _, _, _, edge_mask_hard = k_hop_subgraph(n, 3, G)
        res = groundtruth_metrics(res, gt_grn[edge_mask_hard])
        metrics_nf_grad = [m + r for m, r in zip(metrics_nf_grad, res)]
metrics_nf_grad = [m / (y.shape[0] * X.shape[0]) for m in metrics_nf_grad]
print('=' * 20 + "NFG Results" + '=' * 20)
print({n: v for n, v in zip(m_names, metrics_nf_grad)})


# GNN Explainer
print('=' * 20 + "GNN Explainer" + '=' * 20)
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
final_gnnexp_explanation = torch.zeros_like(gt_grn)
avg_gnnexp_explanation = torch.zeros_like(gt_grn).float()
for x in range(y.shape[0]):
    gnn_exp_explanation = gnn_explainer(X[:, x:x + 1], G)
    final_gnnexp_explanation = torch.max(final_gnnexp_explanation, gnn_exp_explanation.edge_mask)
    avg_gnnexp_explanation += gnn_exp_explanation.edge_mask
    res = groundtruth_metrics(gnn_exp_explanation.edge_mask, gt_grn)
    metrics_gnn_exp = [m + r for m, r in zip(metrics_gnn_exp, res)]
metrics_gnn_exp = [m / y.shape[0] for m in metrics_gnn_exp]
avg_gnnexp_explanation /= y.shape[0]
print('=' * 20 + "GNNExp Results" + '=' * 20)
print({n: v for n, v in zip(m_names, metrics_gnn_exp)})
print({n: v for n, v in zip(m_names, groundtruth_metrics(final_gnnexp_explanation, gt_grn))})
print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_gnnexp_explanation, gt_grn))})
save_masks("gnnexp_max_mask", gt_grn, final_gnnexp_explanation, G)
save_masks("gnnexp_avg_mask", gt_grn, avg_gnnexp_explanation, G)


# PG Explainers
print('=' * 20 + "PG Explainer" + '=' * 20)
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
final_pgexp_explanation = torch.zeros_like(gt_grn)
avg_pgexp_explanation = torch.zeros_like(gt_grn).float()
for x in range(y.shape[0]):
    pg_exp_explanation = pg_explainer(X[:, x:x + 1], G, target=y[x])
    final_pgexp_explanation = torch.max(final_pgexp_explanation, pg_exp_explanation.edge_mask)
    avg_pgexp_explanation += pg_exp_explanation.edge_mask
    res = groundtruth_metrics(pg_exp_explanation.edge_mask, gt_grn)
    metrics_gnn_exp = [m + r for m, r in zip(metrics_gnn_exp, res)]
metrics_gnn_exp = [m / y.shape[0] for m in metrics_gnn_exp]
avg_pgexp_explanation /= y.shape[0]
print('=' * 20 + "PGExp Results" + '=' * 20)
print({n: v for n, v in zip(m_names, metrics_gnn_exp)})
print({n: v for n, v in zip(m_names, groundtruth_metrics(final_pgexp_explanation, gt_grn))})
print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_pgexp_explanation, gt_grn))})
save_masks("pgexp_max_mask", gt_grn, final_pgexp_explanation, G)
save_masks("pgexp_avg_mask", gt_grn, avg_pgexp_explanation, G)
