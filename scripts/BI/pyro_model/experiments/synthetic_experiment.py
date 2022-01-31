import sys
sys.path.append("/users/spate116/singhlab/GCN_Integration/scripts/BI/pyro_model/model")

import numpy as np

from Experiment import Experiment
from samplers.BetaBernoulliSampler import BetaBernoulliSampler
from samplers.NFSampler import NFSampler
from samplers.SpikeSlabSampler import SpikeSlabSampler
from samplers.RandomWalkSampler import RandomWalkSampler

experiment = Experiment("syn3-full-verified", "..")
experiment.train_base_model()
predicate = lambda x: True
label_transform = lambda x, _: x # lambda x, node: x if node < 511 else np.abs(1 - x)

print("Trained Base Model")

rw_hparams = {
    "name": "random_walk",
    "p": 0.25
}
rw_sampler = RandomWalkSampler(**rw_hparams)
experiment.test_sampler(rw_sampler, Experiment.experiment_name(rw_hparams), predicate, label_transform, epochs=10000, lr=0.5, window=500)

print("Finished RW Sampler")

nf_hparams = {
    "name": "normalizing_flows",
    "splines": 12,
    "sigmoid": True,
    "lambd": 5.0,
    "p": 1.5,
}
nf_sampler = NFSampler(device=experiment.device, **nf_hparams)
experiment.test_sampler(nf_sampler, Experiment.experiment_name(nf_hparams), predicate, label_transform, epochs=2000, lr=0.5, window=500)

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
experiment.test_sampler(ss_sampler, Experiment.experiment_name(ss_hparams), predicate, label_transform, epochs=10000, lr=0.05, window=500)

print("Finished SS Sampler")