import pyro
import torch
import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist

from BaseSampler import BaseSampler


class NFSampler(BaseSampler):
    def __init__(self, name, N: int, splines: int, sigmoid: bool, device: torch.device):
        super().__init__(name)
        self.base_dist = dist.Normal()
