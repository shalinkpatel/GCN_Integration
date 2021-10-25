import sys
sys.path.append("../model")

from Experiment import Experiment
from samplers.BetaBernoulliSampler import BetaBernoulliSampler
from samplers.NFSampler import NFSampler
from samplers.SpikeSlabSampler import SpikeSlabSampler

experiment = Experiment("syn3", "..")
experiment.train_base_model()
predicate = lambda x: x >= 512

print("Trained Base Model")

bb_hparams = {
    "name": "beta_bernoulli",
    "alpha": 2.0,
    "beta": 10.0
}
bb_sampler = BetaBernoulliSampler(**bb_hparams)
experiment.test_sampler(bb_sampler, Experiment.experiment_name(bb_hparams), predicate, epochs=2500, lr=0.05, window=500)

print("Finished BetaBernoulli Sampler")

nf_hparams = {
    "name": "normalizing_flows",
    "splines": 12,
    "sigmoid": True,
    "lambd": 5.0,
    "p": 1.5,
}
nf_sampler = NFSampler(device=experiment.device, **nf_hparams)
experiment.test_sampler(nf_sampler, Experiment.experiment_name(nf_hparams), predicate, epochs=2000, lr=0.5, window=500)

print("Finished NF Sampler")

ss_hparams = {
    "name": "spike_slab",
    "theta": 0.25,
    "alpha1": 1.0,
    "beta1": 5.0,
    "alpha2": 10.0,
    "beta2": 1.0
}
ss_sampler = SpikeSlabSampler(**ss_hparams)
experiment.test_sampler(ss_sampler, Experiment.experiment_name(ss_hparams), predicate, epochs=10000, lr=0.05, window=500)

print("Finished SS Sampler")

