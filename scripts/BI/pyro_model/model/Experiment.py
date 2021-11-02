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
import traceback
from datasets.dataset_loaders import load_dataset
from datasets.ground_truth_loaders import load_dataset_ground_truth

class Net(torch.nn.Module):
    def __init__(self, y, x=64):
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
    def __init__(self, experiment: str, base: str, k: int = 3, hidden: int = 64):
        self.experiment = experiment.split('-')[0]
        self.k = k
        edge_index, x, y, _, _, _ = load_dataset(self.experiment, shuffle=False)
        (_, labels), _ = load_dataset_ground_truth(self.experiment)
        
        self.data = Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), y=torch.tensor(y))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.data.to(self.device)
        self.x, self.edge_index = self.data.x, self.data.edge_index
        self.labels = labels

        self.model = Net(self.data.y, x=hidden).to(self.device)

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
                                       torch.mean((torch.argmax(log_logits, dim=1) == self.data.y).float()).item(), epoch)

    def test_sampler(self, sampler: BaseSampler, name: str, predicate=(lambda x: True), label_transform=(lambda x, node: x), **train_hparams):
        auc = 0
        done = 1
        masks = []
        for n in filter(predicate, range(self.x.shape[0])):
            try:
                pyro.clear_param_store()
                node_exp = BayesExplainer(self.model, sampler, n, self.k, self.x, self.data.y, self.edge_index)
                node_exp.train(log=False, **train_hparams)
                edge_mask = node_exp.edge_mask()
                masks += edge_mask.cpu().detach().numpy().tolist()

                self.writer.add_histogram(f"{name}-edge-mask", edge_mask, n)
                self.writer.add_histogram(f"{name}-edge-mask-cum", torch.tensor(masks), n)

                labs = self.labels[node_exp.edge_mask_hard]
                labs = label_transform(labs, n)
                itr_auc = roc_auc_score(labs, edge_mask.cpu().detach().numpy(), average="weighted")
                auc += itr_auc
                done += 1

                ax, _ = node_exp.visualize_subgraph()
                self.writer.add_figure("Importance Graph", ax.get_figure(), n)
                ax, _ = node_exp.visualize_subgraph(edge_mask=labs)
                self.writer.add_figure("Ground Truth Graph", ax.get_figure(), n)

                self.writer.add_scalar(f"{name}-itr-auc", itr_auc, n)
                self.writer.add_scalar(f"{name}-avg-auc", auc / done, n)
            except Exception as e:
                print(f"Encountered an error on node {n} with following error: {e.__str__()}")
                traceback.print_exc()

    @staticmethod
    def experiment_name(hparams: dict) -> str:
        name = ""
        for k, v in hparams.items():
            name += f"{k}-{v}||"
        return name[:-2]
