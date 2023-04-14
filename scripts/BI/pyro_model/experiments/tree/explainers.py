import torch
from tree_model import get_or_train_model
from tqdm import tqdm
import time
from random import shuffle

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric.basic import groundtruth_metrics

from model.DNFGExplainer import DNFGExplainer


# Definitions
m_names = ["accuracy", "recall", "precision", "f1_score", "auroc"]


def save_masks(name: str, grn: torch.Tensor, exp: torch.Tensor, ei: torch.Tensor):
    with open(f'experiments/masks/{name}_tree.csv', 'w') as f:
        f.write('s,d,grn,exp\n')
        for s, d, g, e in zip(ei[0, :], ei[1, :], grn, exp):
            f.write(f'{s.item()},{d.item()},{g.item()},{e.item()}\n')


# Model Loading
model, X, y, G, gt_grn = get_or_train_model()
print('=' * 20 + " DNFG Explainer " + '=' * 20)
metrics_dnf_grad = [0, 0, 0, 0, 0]
samples = list(range(y.shape[0]))
shuffle(samples)
n_samples = 0
graph = 0
final_dnfgexp_explanation = torch.zeros_like(gt_grn).float()
avg_dnfgexp_explanation = torch.zeros_like(gt_grn).float()
for x in samples[:int(0.75 * len(samples))]:
    graph += 1
    start = time.time()
    explainer = DNFGExplainer(model, 64, X[:, x:x + 1], G, device)
    explainer.train(750, 1e-3)
    explainer_mask = explainer.edge_mask().detach()
    explainer.clean()
    del explainer
    if torch.isnan(explainer_mask).sum() == explainer_mask.shape[0]:
        del explainer_mask
        continue
    final_dnfgexp_explanation = torch.max(final_dnfgexp_explanation, explainer_mask)
    avg_dnfgexp_explanation += explainer_mask
    res = groundtruth_metrics(explainer_mask, gt_grn)
    del explainer_mask
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
for x in tqdm(range(y.shape[0])):
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