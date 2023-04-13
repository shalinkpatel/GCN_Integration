import torch
from tree_model import get_or_train_model
from tqdm import tqdm

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric.basic import groundtruth_metrics


# Definitions
m_names = ["accuracy", "recall", "precision", "f1_score", "auroc"]


def save_masks(name: str, grn: torch.Tensor, exp: torch.Tensor, ei: torch.Tensor):
    with open(f'experiments/masks/{name}_tree.csv', 'w') as f:
        f.write('s,d,grn,exp\n')
        for s, d, g, e in zip(ei[0, :], ei[1, :], grn, exp):
            f.write(f'{s.item()},{d.item()},{g.item()},{e.item()}\n')


# Model Loading
model, X, y, G, gt_grn = get_or_train_model()
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
