import torch
from tree_model import get_or_train_model
import time
from random import shuffle

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric.basic import groundtruth_metrics

from model.DNFGExplainer import DNFGExplainer
from model.BetaExplainer import BetaExplainer


# Definitions
m_names = ["accuracy", "recall", "precision", "f1_score", "auroc"]


def save_masks(name: str, grn: torch.Tensor, exp: torch.Tensor, ei: torch.Tensor):
    with open(f'experiments/masks/{name}_tree.csv', 'w') as f:
        f.write('s,d,grn,exp\n')
        for s, d, g, e in zip(ei[0, :], ei[1, :], grn, exp):
            f.write(f'{s.item()},{d.item()},{g.item()},{e.item()}\n')


# Model Loading
device = torch.device('cuda')
model, X, y, G, gt_grn = get_or_train_model(device)
X = X[:, y == 1]
y = y[y == 1]

print('=' * 20 + " DNFG Explainer " + '=' * 20)
metrics_dnf_grad = [0, 0, 0, 0, 0]
samples = list(range(y.shape[0]))
shuffle(samples)
n_samples = 0
graph = 0
final_dnfgexp_explanation = torch.zeros_like(gt_grn).float()
avg_dnfgexp_explanation = torch.zeros_like(gt_grn).float()
for x in samples[:int(0.25 * len(samples))]:
    graph += 1
    start = time.time()
    explainer = DNFGExplainer(model, 8, X[:, x:x + 1], G, device)
    explainer.train(3000, 1e-4)
    print(f"Time for graph {graph}: {time.time() - start}")
    explainer_mask = explainer.edge_mask().detach()
    print(f"Positive Accuracy: {(explainer_mask[gt_grn == 1] > 0.5).float().mean()}")
    print(f"Negative Accuracy: {(explainer_mask[gt_grn == 0] < 0.5).float().mean()}")
    print(f"Edge Mask: {explainer_mask}")
    final_dnfgexp_explanation = torch.max(final_dnfgexp_explanation, explainer_mask)
    avg_dnfgexp_explanation += explainer_mask
    res = groundtruth_metrics(explainer_mask, gt_grn)
    print(f"Graph Result: {res}")
    metrics_dnf_grad = [m + r for m, r in zip(metrics_dnf_grad, res)]
    n_samples += 1
metrics_nf_grad = [m / n_samples for m in metrics_dnf_grad]
avg_dnfgexp_explanation /= n_samples
print('=' * 20 + " DNFG Results " + '=' * 20)
print({n: v for n, v in zip(m_names, metrics_nf_grad)})
print({n: v for n, v in zip(m_names, groundtruth_metrics(final_dnfgexp_explanation, gt_grn))})
print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_dnfgexp_explanation, gt_grn))})
save_masks("dnfgexp_max_mask", gt_grn, final_dnfgexp_explanation, G)
save_masks("dnfgexp_avg_mask", gt_grn, avg_dnfgexp_explanation, G)

print('=' * 20 + " Beta Explainer " + '=' * 20)
metrics_beta = [0, 0, 0, 0, 0]
samples = list(range(y.shape[0]))
shuffle(samples)
n_samples = 0
graph = 0
final_betaexp_explanation = torch.zeros_like(gt_grn).float()
avg_betaexp_explanation = torch.zeros_like(gt_grn).float()
for x in samples[:int(1 * len(samples))]:
    graph += 1
    start = time.time()
    explainer = BetaExplainer(model, X[:, x:x + 1], G, device)
    explainer.train(20000, 1e-4)
    print(f"Time for graph {graph}: {time.time() - start}")
    explainer_mask = explainer.edge_mask().detach()
    print(f"Positive Accuracy: {(explainer_mask[gt_grn == 1] > 0.5).float().mean()}")
    print(f"Negative Accuracy: {(explainer_mask[gt_grn == 0] < 0.5).float().mean()}")
    print(f"Edge Mask: {explainer_mask}")
    final_betaexp_explanation = torch.max(final_betaexp_explanation, explainer_mask)
    avg_betaexp_explanation += explainer_mask
    res = groundtruth_metrics(explainer_mask, gt_grn)
    print(f"Graph Result: {res}")
    metrics_beta = [m + r for m, r in zip(metrics_beta, res)]
    n_samples += 1
metrics_beta = [m / n_samples for m in metrics_beta]
avg_betaexp_explanation /= n_samples
print('=' * 20 + " DNFG Results " + '=' * 20)
print({n: v for n, v in zip(m_names, metrics_beta)})
print({n: v for n, v in zip(m_names, groundtruth_metrics(final_betaexp_explanation, gt_grn))})
print({n: v for n, v in zip(m_names, groundtruth_metrics(avg_betaexp_explanation, gt_grn))})
save_masks("betaexp_max_mask", gt_grn, final_betaexp_explanation, G)
save_masks("betaexp_avg_mask", gt_grn, avg_betaexp_explanation, G)

print('=' * 20 + " GNN Explainer " + '=' * 20)
metrics_gnn_exp = [0, 0, 0, 0, 0]
gnn_explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=500),
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
    start = time.time()
    gnn_exp_explanation = gnn_explainer(X[:, x:x + 1], G)
    print(f"Time for graph {x}: {time.time() - start}")
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
