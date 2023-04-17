from model.Experiment import Experiment
from model.samplers.RandomWalkSampler import RandomWalkSampler
from model.searchers.GNNExplainerSearcher import GNNExplainerSearcher
from model.samplers.NFGradSampler import NFGradSampler
from model.DeterministicExplainer import DeterministicExplainer
from model.BayesExplainer import BayesExplainer
from model.samplers.BaseSampler import BaseSampler
from model.BetaExplainer import BetaExplainer

from torch_geometric.utils import k_hop_subgraph
from loguru import logger

import pyro
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
import numpy as np


def edge_list_to_edge_index(fl: str) -> torch.Tensor:
    with open(fl, "r") as f:
        lines = f.readlines()

    return torch.tensor(list(map(lambda ln: list(map(lambda x: int(x), ln.split(','))), lines))).T


def is_noise(experiment: Experiment, edge_index: torch.Tensor) -> torch.Tensor:
    base_ei = experiment.edge_index
    all_edges = set()
    for i in range(base_ei.shape[1]):
        all_edges.add((base_ei[0, i].item(), base_ei[1, i].item()))

    label = []

    for i in range(edge_index.shape[1]):
        if (edge_index[0, i].item(), edge_index[1, i].item()) in all_edges:
            label.append(1)
        else:
            label.append(0)

    return torch.tensor(label)


def get_exp_sampler(experiment: Experiment, edge_index, sampler, backend, n, **train_hparams) -> torch.Tensor:
    node_exp = backend(experiment.model, sampler, n, experiment.k, experiment.x, experiment.data.y, edge_index)
    node_exp.train(log=False, **train_hparams)
    return node_exp.edge_mask(), node_exp


def get_exp_searcher(experiment: Experiment, edge_index, searcher, backend, n, **train_hparams) -> torch.Tensor:
    node_exp = backend(experiment.model, searcher, n, experiment.k, experiment.x, experiment.data.y, edge_index)
    edge_mask = node_exp.edge_mask('.', **train_hparams)
    return edge_mask, node_exp


def test_sampler(experiment: Experiment, edge_index: torch.Tensor, labels: torch.Tensor, sampler, backend, name: str, **train_hparams):
    logger.info(f"Testing sampler {name}")

    auc = 0
    acc = 0
    aupr = 0
    f1 = 0
    itr_aucs = []
    itr_accs = []
    itr_auprs = []
    itr_f1s = []

    done = 0
    nodes = list(range(experiment.x.shape[0]))

    for n in nodes:
        try:
            pyro.clear_param_store()
            get_exp = get_exp_sampler if isinstance(sampler, BaseSampler) else get_exp_searcher
            edge_mask, node_exp = get_exp(experiment, edge_index, sampler, backend, n, **train_hparams)

            logger.info(edge_mask)

            labs = labels[node_exp.edge_mask_hard]

            for i, v in enumerate(labs.cpu().detach().numpy().tolist()):
                if v == 1:
                    edge_mask[i] = 1

            itr_auc = roc_auc_score(labs, edge_mask.cpu().detach().numpy(), average="weighted")
            itr_acc = accuracy_score(labs, np.array(list(map(lambda x: 0 if x <= 0.5 else 1, edge_mask.detach().cpu().numpy()))))
            itr_aupr = average_precision_score(labs, edge_mask.cpu().detach().numpy(), average="weighted")
            itr_f1 = f1_score(labs, np.array(list(map(lambda x: 0 if x <= 0.5 else 1, edge_mask.detach().cpu().numpy()))))

            itr_aucs.append(itr_auc)
            itr_accs.append(itr_acc)
            itr_auprs.append(itr_aupr)
            itr_f1s.append(itr_f1)

            auc += itr_auc
            acc += itr_acc
            aupr += itr_aupr
            f1 += itr_f1

            done += 1

            logger.info(f"{name.replace('||', '.')} | {n} | itr_auc {itr_auc}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_auc {auc / done}")
            logger.info(f"{name.replace('||', '.')} | {n} | itr_acc {itr_acc}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_acc {acc / done}")
            logger.info(f"{name.replace('||', '.')} | {n} | itr_aupr {itr_aupr}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_aupr {aupr / done}")
            logger.info(f"{name.replace('||', '.')} | {n} | itr_f1 {itr_f1}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_f1 {f1 / done}")
        except ValueError:
            logger.info(f"Skipping node {n} because there is only one class present in labels")

    logger.info(f"{name.replace('||', '.')} | FINISHED | avg_auc {auc / done}")
    logger.info(f"{name.replace('||', '.')} | FINISHED | avg_acc {acc / done}")
    logger.info(f"{name.replace('||', '.')} | FINSIHED | avg_aupr {aupr / done}")
    logger.info(f"{name.replace('||', '.')} | FINISHED | avg_f1 {f1 / done}")


def test_new_explainer(experiment: Experiment, edge_index: torch.Tensor, labels: torch.Tensor, name: str):
    logger.info(f"Testing sampler {name}")

    auc = 0
    acc = 0
    aupr = 0
    f1 = 0
    itr_aucs = []
    itr_accs = []
    itr_auprs = []
    itr_f1s = []

    done = 0
    nodes = list(range(experiment.x.shape[0]))

    X = experiment.x
    k = experiment.k

    for n in nodes:
        try:
            pyro.clear_param_store()
            subset, edge_index_adj, mapping, edge_mask_hard = k_hop_subgraph(n, k, edge_index, relabel_nodes=True)
            X_adj = X[subset]
            explainer = BetaExplainer(experiment.model, X_adj, edge_index, torch.device('cpu'))
            explainer.train(20000, 1e-4)
            edge_mask = explainer.edge_mask()

            labs = labels[edge_mask_hard]

            for i, v in enumerate(labs.cpu().detach().numpy().tolist()):
                if v == 1:
                    edge_mask[i] = 1

            itr_auc = roc_auc_score(labs, edge_mask.cpu().detach().numpy(), average="weighted")
            itr_acc = accuracy_score(labs, np.array(list(map(lambda x: 0 if x <= 0.5 else 1, edge_mask.detach().cpu().numpy()))))
            itr_aupr = average_precision_score(labs, edge_mask.cpu().detach().numpy(), average="weighted")
            itr_f1 = f1_score(labs, np.array(list(map(lambda x: 0 if x <= 0.5 else 1, edge_mask.detach().cpu().numpy()))))

            itr_aucs.append(itr_auc)
            itr_accs.append(itr_acc)
            itr_auprs.append(itr_aupr)
            itr_f1s.append(itr_f1)

            auc += itr_auc
            acc += itr_acc
            aupr += itr_aupr
            f1 += itr_f1

            done += 1

            logger.info(f"{name.replace('||', '.')} | {n} | itr_auc {itr_auc}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_auc {auc / done}")
            logger.info(f"{name.replace('||', '.')} | {n} | itr_acc {itr_acc}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_acc {acc / done}")
            logger.info(f"{name.replace('||', '.')} | {n} | itr_aupr {itr_aupr}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_aupr {aupr / done}")
            logger.info(f"{name.replace('||', '.')} | {n} | itr_f1 {itr_f1}")
            logger.info(f"{name.replace('||', '.')} | {n} | avg_f1 {f1 / done}")
        except ValueError:
            logger.info(f"Skipping node {n} because there is only one class present in labels")


# --------------- Setting Up Data -----------------
experiment = Experiment("syn3-full-verifed", "..")
experiment.train_base_model()

logger.info("Finished training base model")

noisy_G = edge_list_to_edge_index("/users/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/data/sanity/noisy.data")
gr_truth = is_noise(experiment, noisy_G)

# --------------- BetaExplainer --------------------
test_new_explainer(experiment, noisy_G, gr_truth, "BetaExplainer")
logger.info("Finished evaluating BetaExplainer on Sanity Check Experiment")
exit()

# --------------- NFGradSampler -----------------
nfg_hparams = {
    "name": "normalizing_flows_grad",
    "splines": 10,
    "sigmoid": True,
    "lambd": 5.0,
    "p": 1.5,
}
nfg_sampler = NFGradSampler(device=experiment.device, **nfg_hparams)
test_sampler(experiment, noisy_G, gr_truth, nfg_sampler, BayesExplainer, Experiment.experiment_name(nfg_hparams), epochs=1500, lr=0.0001, window=500)

logger.info("Finished evaluating NFGradSampler on Sanity Check Dataset")

# --------------- RWSampler ----------------- 
rw_hparams = {
    "name": "random_walk",
    "p": 0.90,
}
rw_sampler = RandomWalkSampler(**rw_hparams)
test_sampler(experiment, noisy_G, gr_truth, rw_sampler, BayesExplainer, Experiment.experiment_name(rw_hparams), epochs=15000, lr=0.0001, window=500)

logger.info("Finished evaluating RWSampler on Sanity Check Dataset")

# --------------- GNNExplainer -----------------
ge_hparams = {
    "name": "gnn_explainer",
    "epochs": 1000
}
ge_searcher = GNNExplainerSearcher(**ge_hparams)
test_sampler(experiment, noisy_G, gr_truth, ge_searcher, DeterministicExplainer, Experiment.experiment_name(ge_hparams))

logger.info("Finished evaluating GNNExplainer on Sanity Check Dataset")


