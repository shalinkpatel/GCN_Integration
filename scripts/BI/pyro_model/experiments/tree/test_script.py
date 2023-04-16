from model.DNFGExplainer import DNFGExplainer
from experiments.tree.tree_model import get_or_train_model
import torch
import numpy as np


model, X, y, G, gt_grn = get_or_train_model()
explainer = DNFGExplainer(model, 8, X[:, 600:601], G, torch.device('cpu'))
explainer.train(3000, 1e-4)
print(f"Positive Accuracy: {(explainer.edge_mask()[gt_grn == 1] > 0.5).float().mean()}")
print(f"Negative Accuracy: {(explainer.edge_mask()[gt_grn == 0] < 0.5).float().mean()}")
print(f"Edge Mask: {explainer.edge_mask()}")
np.save("experiments/tree/example_DNFG_dist_1.npy", explainer.edge_distribution().numpy())

explainer = DNFGExplainer(model, 8, X[:, 601:602], G, torch.device('cpu'))
explainer.train(3000, 1e-4)
print(f"Positive Accuracy: {(explainer.edge_mask()[gt_grn == 1] > 0.5).float().mean()}")
print(f"Negative Accuracy: {(explainer.edge_mask()[gt_grn == 0] < 0.5).float().mean()}")
print(f"Edge Mask: {explainer.edge_mask()}")
np.save("experiments/tree/example_DNFG_dist_2.npy", explainer.edge_distribution().numpy())
