from model.BetaExplainer import BetaExplainer

import torch
import torch.nn.functional as F
from itertools import chain
from random import shuffle
import time

from torch_geometric.nn import GCNConv
from torch_geometric.explain.metric.basic import groundtruth_metrics


# Definitions
class Model(torch.nn.Module):
    def __init__(self, y, N, x=64):
        super(Model, self).__init__()
        self.conv1 = GCNConv(1, 10)
        self.conv2 = GCNConv(10, x)
        self.conv3 = GCNConv(x, x)
        self.fc = torch.nn.Linear(x * N, max(y).tolist() + 1)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones_like(edge_index[0, :]).float()
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_weight))
        return self.fc(x.flatten()).log_softmax(dim=0)


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
    with open(f'experiments/masks/{name}_{groups}.csv', 'w') as f:
        f.write('s,d,grn,exp\n')
        for s, d, g, e in zip(ei[0, :], ei[1, :], grn, exp):
            f.write(f'{s.item()},{d.item()},{g.item()},{e.item()}\n')


# Main Test
device = torch.device('cpu')

groups = 9
X = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-1sparsity.pt").float().to(device)
y = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-labels.pt").to(device)
G = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-1sparsity-compgraph.pt").to(device)
grn = torch.load(
    f"model/sergio/final_data/100gene-{groups}groups"
    f"-gt-grn.pt").to(device)

grn_s = set(
    [(s.cpu().item(), d.cpu().item()) for s, d in zip(chain(grn[0, :], grn[1, :]), chain(grn[1, :], grn[0, :]))])
gt_grn = torch.tensor([1 if (s.cpu().item(), d.cpu().item()) in grn_s else 0 for s, d in zip(G[0, :], G[1, :])]).to(device)

model = Model(y, 100)
print('=' * 20 + " Loading Previous Model " + '=' * 20)
model.load_state_dict(torch.load(f"experiments/models/graph_class_{groups}groups.pt", map_location=torch.device('cpu')))
model = model.to(device)

m_names = ["accuracy", "recall", "precision", "f1_score", "auroc"]

# DNFGSampler
print('=' * 20 + " Beta Explainer " + '=' * 20)
metrics_dnf_grad = [0, 0, 0, 0, 0]
samples = list(range(y.shape[0]))
shuffle(samples)
n_samples = 0
graph = 0
final_betaexp_explanation = torch.zeros_like(gt_grn).float()
avg_betaexp_explanation = torch.zeros_like(gt_grn).float()
for x in samples[:int(0.75 * len(samples))]:
    graph += 1
    start = time.time()
    explainer = BetaExplainer(model, X[:,x:x+1], G, device)
    explainer.train(100000, 0.05)
    print(f"Time for graph {graph}: {time.time() - start}")
    explainer_mask = explainer.edge_mask().detach()
    print(explainer_mask)
    print(explainer_mask.min())
    print(explainer_mask.max())
    print(explainer_mask.mean())
    del explainer
    if torch.isnan(explainer_mask).sum() == explainer_mask.shape[0]:
        del explainer_mask
        continue
    final_betaexp_explanation = torch.max(final_betaexp_explanation, explainer_mask)
    avg_betaexp_explanation += explainer_mask
    res = groundtruth_metrics(explainer_mask, gt_grn)
    del explainer_mask
    print(res)
    metrics_dnf_grad = [m + r for m, r in zip(metrics_dnf_grad, res)]
    n_samples += 1
metrics_nf_grad = [m / n_samples for m in metrics_dnf_grad]
avg_betaexp_explanation /= n_samples
print('=' * 20 + " Beta Results " + '=' * 20)
print({n: v for n, v in zip(m_names, metrics_nf_grad)})
print({n: v for n, v in zip(m_names, groundtruth_metrics(final_betaexp_explanation, gt_grn))})
print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_betaexp_explanation, gt_grn))})
save_masks("betaexp_max_mask", gt_grn, final_betaexp_explanation, G)
save_masks("betaexp_avg_mask", gt_grn, avg_betaexp_explanation, G)
