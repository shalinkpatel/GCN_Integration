import sys

sys.path.append("/users/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/model")

from Experiment import Experiment
from samplers.NFSampler import NFSampler
from samplers.SpikeSlabSampler import SpikeSlabSampler
from samplers.RandomWalkSampler import RandomWalkSampler
from searchers.GNNExplainerSearcher import GNNExplainerSearcher
from searchers.GreedySearcher import GreedySearcher

from loguru import logger


experiment = Experiment("syn3-full-verified", "..")
experiment.train_base_model()
predicate = lambda x: True
label_transform = lambda x, _: x # lambda x, node: x if node < 511 else np.abs(1 - x)

logger.info("Trained Base Model")

gs_hparams = {
    "name": "greedy_searcher",
    "edges": 6
}
gs_searcher = GreedySearcher(**gs_hparams)
experiment.test_sampler(gs_searcher, Experiment.experiment_name(gs_hparams), predicate, label_transform)

logger.info("Finished Greedy Searcher")

ge_hparams = {
    "name": "gnn_explainer",
    "epochs": 1000
}
ge_searcher = GNNExplainerSearcher(**ge_hparams)
experiment.test_sampler(ge_searcher, Experiment.experiment_name(ge_hparams), predicate, label_transform)

logger.info("Finished GNNExp Searcher")

rw_hparams = {
    "name": "random_walk",
    "p": 0.5,
}
rw_sampler = RandomWalkSampler(**rw_hparams)
experiment.test_sampler(rw_sampler, Experiment.experiment_name(rw_hparams), predicate, label_transform, epochs=10000, lr=0.15, window=500)

logger.info("Finished RW Sampler")

nf_hparams = {
    "name": "normalizing_flows",
    "splines": 12,
    "sigmoid": True,
    "lambd": 5.0,
    "p": 1.5,
}
nf_sampler = NFSampler(device=experiment.device, **nf_hparams)
experiment.test_sampler(nf_sampler, Experiment.experiment_name(nf_hparams), predicate, label_transform, epochs=2000, lr=0.5, window=500)

logger.info("Finished NF Sampler")

ss_hparams = {
    "name": "spike_slab",
    "theta": 0.25,
    "alpha1": 1.0,
    "beta1": 5.0,
    "alpha2": 10.0,
    "beta2": 1.0
}
ss_sampler = SpikeSlabSampler(**ss_hparams)
experiment.test_sampler(ss_sampler, Experiment.experiment_name(ss_hparams), predicate, label_transform, epochs=10000, lr=0.05, window=500)

logger.info("Finished SS Sampler")
