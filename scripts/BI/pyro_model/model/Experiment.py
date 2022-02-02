from os.path import exists
from os import makedirs
from shutil import rmtree
from glob import glob

import pyro
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from BayesExplainer import BayesExplainer
from DeterministicExplainer import DeterministicExplainer
from samplers.BaseSampler import BaseSampler
from searchers.BaseSearcher import BaseSearcher
import traceback
from datasets.dataset_loaders import load_dataset
from datasets.ground_truth_loaders import load_dataset_ground_truth
from loguru import logger

from utils.serialization import with_serializer

from multipledispatch import dispatch

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


class TestSet:
    def __init__(self, experiment: str):
        self.experiment = experiment
        self.files = glob(f"tests/{self.experiment}/*.pt")
        self.files.sort()
        self.subset = list(map(lambda x: int(x.split("/")[-1].replace(".pt", "")), self.files))
        self.labels = list(map(lambda x: torch.load(x), self.files))

    def get(self, idx: int) -> torch.Tensor:
        return self.labels[self.subset.index(idx)]


class Experiment:
    def __init__(self, experiment: str, base: str, k: int = 3, hidden: int = 64):
        self.experiment = experiment.split('-')[0]
        self.using_test_set = False
        if "verified" in experiment:
            self.using_test_set = True
            self.test_set = TestSet(self.experiment)
        self.k = k
        edge_index, x, y, _, _, _ = load_dataset(self.experiment, shuffle=False)
        (_, labels), _ = load_dataset_ground_truth(self.experiment)
        
        self.data = Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), y=torch.tensor(y))

        self.device = torch.device('cpu')
        self.data = self.data.to(self.device)
        self.x, self.edge_index = self.data.x, self.data.edge_index
        self.labels = labels

        self.model = Net(self.data.y, x=hidden).to(self.device)

        self.base = base
        self.exp_name = experiment
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

    @with_serializer("dev")
    def test_sampler(self, sampler, name: str, predicate=(lambda x: True), label_transform=(lambda x, node: x), **train_hparams):
        auc = 0
        acc = 0
        itr_aucs = []
        itr_accs = []

        path = f"{self.base}/logs/{self.experiment}/"
        if exists(path):
            rmtree(path)
        else:
            makedirs(path)
        logger.add(path + f"{name.replace('||', '.')}.log")

        done = 0
        masks = []
        nodes = set(filter(predicate, range(self.x.shape[0])))
        if self.using_test_set:
            nodes_test_set = set(self.test_set.subset)
            nodes = nodes.intersection(nodes_test_set)
            nodes = list(nodes)
            nodes.sort()
        for n in nodes:
            try:
                pyro.clear_param_store()
                edge_mask = self.get_exp(sampler, n, **train_hparams)
                masks += edge_mask.cpu().detach().numpy().tolist()

                self.writer.add_histogram(f"{name}-edge-mask", edge_mask, n)
                self.writer.add_histogram(f"{name}-edge-mask-cum", torch.tensor(masks), n)

                if not self.using_test_set:
                    labs = self.labels[self.node_exp.edge_mask_hard]
                else:
                    labs = self.test_set.get(n)
                labs = label_transform(labs, n)
                itr_auc = roc_auc_score(labs, edge_mask.cpu().detach().numpy(), average="weighted")
                itr_aucs.append(itr_auc)

                itr_acc = accuracy_score(labs, edge_mask.detach().cpu().numpy() <= 0.5)
                itr_accs.append(itr_acc)
                itr_aucs.append(itr_auc)

                auc += itr_auc
                acc += itr_acc
                done += 1

                ax, _ = self.node_exp.visualize_subgraph()
                self.writer.add_figure("Importance Graph", ax.get_figure(), n)
                ax, _ = self.node_exp.visualize_subgraph(edge_mask=labs)
                self.writer.add_figure("Ground Truth Graph", ax.get_figure(), n)

                self.writer.add_scalar(f"{name}-itr-auc", itr_auc, n)
                self.writer.add_scalar(f"{name}-avg-auc", auc / done, n)
                self.writer.add_scalar(f"{name}-itr-acc", itr_acc, n)
                self.writer.add_scalar(f"{name}-avg-acc", acc / done, n)

                logger.info(f"{name.replace('||', '.')} | {n} | itr_auc {itr_auc}")
                logger.info(f"{name.replace('||', '.')} | {n} | avg_auc {auc / done}")
                logger.info(f"{name.replace('||', '.')} | {n} | itr_acc {itr_acc}")
                logger.info(f"{name.replace('||', '.')} | {n} | avg_acc {acc / done}")
            except Exception as e:
                print(f"Encountered an error on node {n} with following error: {e.__str__()}")
                traceback.print_exc()
            print(f"Analyzed node {n} fully.")

        return name, itr_accs, itr_aucs

    @dispatch(BaseSampler, int)
    def get_exp(self, sampler, n, **train_hparams) -> torch.Tensor:
        node_exp = BayesExplainer(self.model, sampler, n, self.k, self.x, self.data.y, self.edge_index)
        node_exp.train(log=False, **train_hparams)
        self.node_exp = node_exp
        return node_exp.edge_mask()
    
    @dispatch(BaseSearcher, int)
    def get_exp(self, searcher, n, **train_hparams) -> torch.Tensor:
        node_exp = DeterministicExplainer(self.model, searcher, n, self.k, self.x, self.data.y, self.edge_index)
        edge_mask = node_exp.edge_mask('.', **train_hparams)
        self.node_exp = node_exp
        return edge_mask

    @staticmethod
    def experiment_name(hparams: dict) -> str:
        name = ""
        for k, v in hparams.items():
            name += f"{k}-{v}||"
        return name[:-2]
