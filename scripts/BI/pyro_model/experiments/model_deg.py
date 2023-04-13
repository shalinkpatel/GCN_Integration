import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from itertools import chain
import pandas as pd


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
        return self.fc(x.flatten()).log_softmax(dim=0)


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
model.eval()
model = model.to(device)


# Load Testing
df = pd.DataFrame(columns=["sparsity", "acc"])
for i in range(11):
    sparsity = (i / 10) * torch.ones(G.shape[1]).float()
    set_masks(model, sparsity, G, False)

    correct = 0
    for n in range(y.shape[0]):
        log_logits = model(X[:, n:n + 1], G)
        correct += (torch.argmax(log_logits) == y[n].item()).float().item()
    acc = correct / y.shape[0]
    df.loc[len(df.index)] = (i / 10, acc)
    clear_masks(model)
df.to_csv('experiments/deg.csv')
