from tree_model import get_or_train_model
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
import pandas as pd
import torch


model, X, y, G, _ = get_or_train_model()

df = pd.DataFrame(columns=["sparsity", "acc"])
for i in range(11):
    sparsity = (i / 10) * torch.ones(G.shape[1]).float()
    set_masks(model, sparsity, G, False)

    correct = 0
    for n in range(y.shape[0]):
        log_logits = model(X[:, n:n + 1], G)
        probs = log_logits.softmax(dim=1)
        correct += (torch.argmax(probs) == y[n].item()).float().item()
    acc = correct / y.shape[0]
    df.loc[len(df.index)] = (i / 10, acc)
    clear_masks(model)
df.to_csv('experiments/tree/deg.csv', index=False)
