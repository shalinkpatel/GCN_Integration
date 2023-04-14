from tree_model import get_or_train_model
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
import pandas as pd
import torch


model, X, y, G, gt_grn = get_or_train_model()

df = pd.DataFrame(columns=["sparsity", "acc", "graph_type"])
for i in range(11):
    sparsity = (i / 10) * torch.ones(G.shape[1]).float()
    set_masks(model, sparsity, G, False)

    correct = 0
    for n in range(y.shape[0]):
        log_logits = model(X[:, n:n + 1], G)
        probs = log_logits.softmax(dim=1)
        correct += (torch.argmax(probs) == y[n].item()).float().item()
    acc = correct / y.shape[0]
    df.loc[len(df.index)] = (i / 10, acc, "whole")
    clear_masks(model)

for i in range(11):
    set_masks(model, gt_grn.float() * (i / 10), G, False)
    correct = 0
    for n in range(y.shape[0]):
        log_logits = model(X[:, n:n + 1], G)
        probs = log_logits.softmax(dim=1)
        correct += (torch.argmax(probs) == y[n].item()).float().item()
    acc = correct / y.shape[0]
    clear_masks(model)
    df.loc[len(df.index)] = (i / 10, acc, "gt")
    print(f"GRN Acc @ sparsity {i / 10}: {acc}")


for i in range(11):
    set_masks(model, (1 - gt_grn.float()) * (i / 10), G, False)
    correct = 0
    for n in range(y.shape[0]):
        log_logits = model(X[:, n:n + 1], G)
        probs = log_logits.softmax(dim=1)
        correct += (torch.argmax(probs) == y[n].item()).float().item()
    acc = correct / y.shape[0]
    clear_masks(model)
    df.loc[len(df.index)] = (i / 10, acc, "non-gt")
    print(f"Non-GRN Acc @ sparsity {i / 10}: {acc}")

df.to_csv("experiments/tree/deg.csv", index=False)