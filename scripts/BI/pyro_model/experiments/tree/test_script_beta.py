from model.BetaExplainer import BetaExplainer
from experiments.tree.tree_model import get_or_train_model
from torch_geometric.explain.metric.basic import groundtruth_metrics
import torch
import numpy as np


model, X, y, G, gt_grn = get_or_train_model()
explainer = BetaExplainer(model, X[:, 600:601], G, torch.device('cpu'))
explainer.train(20000, 1e-4)
print(f"Positive Accuracy: {(explainer.edge_mask()[gt_grn == 1] > 0.5).float().mean()}")
print(f"Negative Accuracy: {(explainer.edge_mask()[gt_grn == 0] < 0.5).float().mean()}")
print(f"Edge Mask: {explainer.edge_mask()}")
print(f"Metrics: {groundtruth_metrics(explainer.edge_mask(), gt_grn)}")
np.save("experiments/tree/example_beta_dist_1.npy", explainer.edge_distribution().numpy())

explainer = BetaExplainer(model, X[:, 601:602], G, torch.device('cpu'))
explainer.train(20000, 1e-4)
print(f"Positive Accuracy: {(explainer.edge_mask()[gt_grn == 1] > 0.5).float().mean()}")
print(f"Negative Accuracy: {(explainer.edge_mask()[gt_grn == 0] < 0.5).float().mean()}")
print(f"Edge Mask: {explainer.edge_mask()}")
print(f"Metrics: {groundtruth_metrics(explainer.edge_mask(), gt_grn)}")
np.save("experiments/tree/example_beta_dist_2.npy", explainer.edge_distribution().numpy())
