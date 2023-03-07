import sys

sys.path.append("/users/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/model")

from Experiment import Experiment
from samplers.NFSampler import NFSampler
from samplers.SpikeSlabSampler import SpikeSlabSampler
from samplers.RandomWalkSampler import RandomWalkSampler
from searchers.GNNExplainerSearcher import GNNExplainerSearcher
from searchers.GreedySearcher import GreedySearcher
from samplers.BetaGradSampler import BetaGradSampler
from samplers.NFGradSampler import NFGradSampler
from samplers.SpikeSlabGradSampler import SpikeSlabGradSampler
from DeterministicExplainer import DeterministicExplainer
from BayesExplainer import BayesExplainer
from MCMCExplainer import MCMCExplainer

from loguru import logger


experiment = Experiment("syn3-full-verifed", "..")
experiment.train_base_model()
predicate = lambda x: True
label_transform = lambda x, _: x # lambda x, node: x if node < 511 else np.abs(1 - x)

logger.info("Trained Base Model")

gs_hparams = {
    "name": "greedy_searcher",
    "edges": 6
}
gs_searcher = GreedySearcher(**gs_hparams)
experiment.test_sampler(gs_searcher, DeterministicExplainer, Experiment.experiment_name(gs_hparams), predicate, label_transform)

logger.info("Finished Greedy Searcher")

ge_hparams = {
    "name": "gnn_explainer",
    "epochs": 1000
}
ge_searcher = GNNExplainerSearcher(**ge_hparams)
experiment.test_sampler(ge_searcher, DeterministicExplainer, Experiment.experiment_name(ge_hparams), predicate, label_transform)

logger.info("Finished GNNExp Searcher")

bg_hparams = {
    "name": "beta_grad_mcmc",
    "alpha": 2.0,
    "beta": 5.0
}
bg_sampler = BetaGradSampler(**bg_hparams)
experiment.test_sampler(bg_sampler, MCMCExplainer, Experiment.experiment_name(bg_hparams), predicate, label_transform)

logger.info("Finished BetaGrad MCMC Sampler")

ssg_hparams = {
    "name": "spike_slab_grad_mcmc",
    "alpha1": 1.0,
    "beta1": 5.0,
}
ssg_sampler = SpikeSlabGradSampler(**ssg_hparams)
experiment.test_sampler(ssg_sampler, MCMCExplainer, Experiment.experiment_name(ssg_hparams), predicate, label_transform, epochs=10000, lr=0.0001, window=500)

logger.info("Finished SSGrad Sampler")

bg_hparams = {
    "name": "beta_grad",
    "alpha": 2,
    "beta": 5
}
bg_sampler = BetaGradSampler(**bg_hparams)
experiment.test_sampler(bg_sampler, BayesExplainer, Experiment.experiment_name(bg_hparams), predicate, label_transform)

logger.info("Finished BetaGrad Sampler")

nfg_hparams = {
    "name": "normalizing_flows_grad",
    "splines": 12,
    "sigmoid": True,
    "lambd": 5.0,
    "p": 1.5,
}
nfg_sampler = NFGradSampler(device=experiment.device, **nfg_hparams)
experiment.test_sampler(nfg_sampler, BayesExplainer, Experiment.experiment_name(nfg_hparams), predicate, label_transform, epochs=2000, lr=0.0001, window=500)

logger.info("Finished NFGrad Sampler")

ssg_hparams = {
    "name": "spike_slab_grad",
    "alpha1": 1.0,
    "beta1": 5.0,
}
ssg_sampler = SpikeSlabGradSampler(**ssg_hparams)
experiment.test_sampler(ssg_sampler, BayesExplainer, Experiment.experiment_name(ssg_hparams), predicate, label_transform, epochs=10000, lr=0.0001, window=500)

logger.info("Finished SSGrad Sampler")

rw_hparams = {
    "name": "random_walk",
    "p": 0.1,
}
rw_sampler = RandomWalkSampler(**rw_hparams)
experiment.test_sampler(rw_sampler, BayesExplainer, Experiment.experiment_name(rw_hparams), predicate, label_transform, epochs=15000, lr=0.0001, window=500)

logger.info("Finished RW Sampler")

nf_hparams = {
    "name": "normalizing_flows",
    "splines": 12,
    "sigmoid": True,
    "lambd": 5.0,
    "p": 1.5,
}
nf_sampler = NFSampler(device=experiment.device, **nf_hparams)
experiment.test_sampler(nf_sampler, BayesExplainer, Experiment.experiment_name(nf_hparams), predicate, label_transform, epochs=2000, lr=0.0001, window=500)

logger.info("Finished NF Sampler")

ss_hparams = {
    "name": "spike_slab",
    "alpha1": 1.0,
    "beta1": 5.0,
}
ss_sampler = SpikeSlabSampler(**ss_hparams)
experiment.test_sampler(ss_sampler, BayesExplainer, Experiment.experiment_name(ss_hparams), predicate, label_transform, epochs=10000, lr=0.0001, window=500)

logger.info("Finished SS Sampler")
