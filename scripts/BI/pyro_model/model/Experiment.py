from os.path import exists
from shutil import rmtree

import pyro
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle
import networkx as nx

from BayesExplainer import BayesExplainer
from samplers.BaseSampler import BaseSampler


class Net(torch.nn.Module):
    def __init__(self, x=64):
        super(Net, self).__init__()
        self.conv1 = GCNConv(10, x)
        self.conv2 = GCNConv(x, x)
        self.conv3 = GCNConv(x, x)
        self.fc = torch.nn.Linear(x, max(y).tolist() + 1)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        return self.fc(x)


class Experiment:
    def __init__(self, prefix: str, experiment: str, base: str, k: int = 3, hidden: int = 64):
        self.prefix = prefix
        self.experiment = experiment
        self.G = nx.read_gpickle(prefix + f'pyro_model/synthetic/data/{self.experiment}_G.pickle')
        self.k = k
        with open(prefix + f'pyro_model/synthetic/data/{self.experiment}_lab.pickle', 'rb') as f:
            self.labels = pickle.load(f)

        with open(prefix + f'PGExplainer/dataset/{self.experiment}.pkl', 'rb') as f:
            adj, _, _, _, _, _, _, _, edge_labels = pickle.load(f)
        self.edge_labels = torch.tensor(edge_labels)

        self.x = torch.tensor([x[1]['feat'] for x in self.G.nodes(data=True)])
        self.edge_index = torch.tensor([x for x in self.G.edges])
        self.edge_index_flipped = self.edge_index[:, [1, 0]]
        self.edge_index = torch.cat((self.edge_index, self.edge_index_flipped))
        self.y = torch.tensor(self.labels, dtype=torch.long)
        self.data = Data(x=self.x, edge_index=self.edge_index.T, y=self.y)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.data.to(self.device)
        self.x, self.edge_index = self.data.x, self.data.edge_index

        self.model = Net(x=hidden).to(self.device)

        path = f"{base}/runs/{experiment}"
        if exists(path):
            rmtree(path)
        self.writer = SummaryWriter(path)

    def train_base_model(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        best_loss = 100
        pbar = range(10000)
        for epoch in pbar:
            # Training step
            self.model.train()
            optimizer.zero_grad()
            log_logits = self.model(self.x, self.edge_index)
            loss = F.cross_entropy(log_logits, self.data.y)
            loss.backward()
            optimizer.step()

            # Testing step
            self.model.eval()
            best_loss = loss if loss < best_loss else best_loss
            if epoch % 100 == 0:
                self.writer.add_scalar("GNN Acc",
                                       torch.mean((torch.argmax(log_logits, dim=1) == self.data.y).float()).item())

    def test_sampler(self, sampler: BaseSampler, name: str, **train_hparams):
        auc = 0
        done = 1
        masks = []
        for n in range(self.x.shape[0]):
            try:
                pyro.clear_param_store()
                node_exp = BayesExplainer(self.model, sampler, n, self.k, self.x, self.data.y, self.edge_index)
                node_exp.train(log=False, **train_hparams)
                edge_mask = node_exp.edge_mask()
                masks += edge_mask.cpu().detach().numpy().tolist()

                self.writer.add_histogram(f"{name}-edge-mask", edge_mask, n)
                self.writer.add_histogram(f"{name}-edge-mask-cum", masks, n)

                edges = node_exp.edge_index_adj
                labs = self.edge_labels[node_exp.subset, :][:, node_exp.subset][edges[0, :], edges[1, :]]
                sub_idx = (labs.long().cpu().detach().numpy() == 1)
                itr_auc = roc_auc_score(labs.long().cpu().detach().numpy()[sub_idx],
                                        edge_mask.cpu().detach().numpy()[sub_idx])
                auc += itr_auc
                done += 1

                self.writer.add_scalar(f"{name}-itr-auc", itr_auc, n)
                self.writer.add_scalar(f"{name}-avg-auc", auc / done, n)
            except Exception as e:
                print(f"Encountered an error on node {n} with following error: {e.__str__()}")

    @staticmethod
    def experiment_name(hparams: dict) -> str:
        name = ""
        for k, v in hparams.items():
            name += f"{k}-{v}||"
        return name[:-2]









