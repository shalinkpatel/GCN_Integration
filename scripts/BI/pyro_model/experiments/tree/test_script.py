from model.DNFGExplainer import DNFGExplainer
from experiments.tree.tree_model import get_or_train_model
import torch


model, X, y, G, gt_grn = get_or_train_model()
explainer = DNFGExplainer(model, 8, X[:, 600:601], G, torch.device('cpu'))
explainer.train(3000, 1e-4)
print(f"Positive Accuracy: {(explainer.edge_mask()[gt_grn == 1] > 0.5).float().mean()}")
print(f"Negative Accuracy: {(explainer.edge_mask()[gt_grn == 0] < 0.5).float().mean()}")
print(explainer.edge_mask())

explainer = DNFGExplainer(model, 8, X[:, 601:602], G, torch.device('cpu'))
explainer.train(3000, 1e-4)
print(f"Positive Accuracy: {(explainer.edge_mask()[gt_grn == 1] > 0.5).float().mean()}")
print(f"Negative Accuracy: {(explainer.edge_mask()[gt_grn == 0] < 0.5).float().mean()}")
print(explainer.edge_mask())
