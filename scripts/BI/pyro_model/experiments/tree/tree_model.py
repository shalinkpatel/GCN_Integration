import torch
import torch.nn.functional as F
from random import shuffle
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from os.path import exists
from copy import deepcopy
from itertools import chain


class GCN(torch.nn.Module):
    def __init__(self, y, N, x, device):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(1, x)
        self.conv2 = SAGEConv(x, x)
        self.conv3 = SAGEConv(x, x)
        self.fc1 = torch.nn.Linear(x, y.max() + 1)
        self.N = N
        self.device = device

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(self.N).to(self.device).long()

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        return self.fc1(x).log_softmax(dim=1)


def train_model(model, X, y, edge_index, G, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0
    print('=' * 20 + ' Started Training ' + '=' * 20)
    pbar = range(100)
    best_weights = None
    idxs = list(range(y.shape[0]))
    shuffle(idxs)
    for epoch in pbar:
        # Training step
        model.train()
        loss_ep = 0
        avg_max = 0
        data_list = [Data(x=X[:, n:n + 1], y=y[n], edge_index=edge_index) for n in idxs[:int(0.5 * len(idxs))]]
        loader = DataLoader(data_list, batch_size=8)
        for grp in loader:
            optimizer.zero_grad()
            logits = model(grp.x, grp.edge_index, grp.batch)
            loss = F.cross_entropy(logits, grp.y)
            probs = logits.softmax(dim=1)
            avg_max += probs.detach().amax(dim=1).sum().item()
            loss_ep += loss.detach().item()
            loss.backward()
            optimizer.step()
        avg_max /= (0.5 * len(idxs))

        # Testing step
        correct = 0
        for n in idxs[int(0.5 * len(idxs)):]:
            probs = model(X[:, n:n + 1], G).softmax(dim=1)
            correct += (torch.argmax(probs) == y[n].item()).float().item()
        acc = correct / (0.5 * len(idxs))

        best_acc = acc if acc > best_acc else best_acc
        if best_acc == acc:
            model.to(torch.device('cpu'))
            best_weights = deepcopy(model.state_dict())
            model.to(device)
        print(f"Epoch {epoch} | Best Acc = {best_acc} | Acc = {acc} | Loss = {loss_ep / len(idxs)} | Avg Max = {avg_max}")
    print('=' * 20 + ' Ended Training ' + '=' * 20)
    correct = 0
    for n in range(y.shape[0]):
        probs = model(X[:, n:n + 1], edge_index)
        correct += (torch.argmax(probs) == y[n].item()).float().item()
    acc = correct / y.shape[0]
    print(acc)
    return best_weights


def get_or_train_model(device=torch.device('cpu')):
    x = torch.load("experiments/tree/x.pt").to(device)
    y = torch.load("experiments/tree/y.pt").to(device)
    G = torch.load("experiments/tree/comp_graph.pt").to(device)
    grn = torch.load("experiments/tree/gt_grn.pt").to(device)
    grn_s = set([(s.cpu().item(), d.cpu().item()) for s, d in zip(chain(grn[0, :], grn[1, :]), chain(grn[1, :], grn[0, :]))])
    gt_grn = torch.tensor([1 if (s.cpu().item(), d.cpu().item()) in grn_s else 0 for s, d in zip(G[0, :], G[1, :])]).to(device)

    model = GCN(y, 7, 32, device).to(device)
    if exists("experiments/tree/model.pt"):
        print('=' * 20 + " LOADING MODEL " + '=' * 20)
        model.load_state_dict(torch.load("experiments/tree/model.pt", map_location=torch.device('cpu')))
        model.to(device)
    else:
        print('=' * 20 + " TRAINING MODEL " + '=' * 20)
        sd = train_model(model, x, y, grn, G, device)
        torch.save(sd, 'experiments/tree/model.pt')

    return model, x, y, G, gt_grn


if __name__ == '__main__':
    get_or_train_model()
