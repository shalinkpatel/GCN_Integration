import pyro
import torch
import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist
import pyro.distributions.transforms as T

from BaseSampler import BaseSampler


class NFSampler(BaseSampler):
    def __init__(self, name, N: int, splines: int, sigmoid: bool, device: torch.device):
        super().__init__(name)
        self.base_dist = dist.Normal(torch.zeros(N).to(device), torch.ones(N).to(device))
        self.splines = []
        for _ in range(splines):
            self.splines.append(T.spline(N).to(device))
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.splines)
        self.sigmoid = sigmoid

    def sample_model(self, X, y, explainer):
        m_sub = self.flow_dist.rsample(torch.Size([250,]))
        if self.sigmoid:
            m_sub = m_sub.sigmoid().clamp(0, 1).mean(dim=0).to_event(1)
        else:
            m_sub = m_sub.clamp(0, 1).mean(dim=0).to_event(1)
        m = pyro.sample("m", dist.Bernoulli(m_sub))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1)
        y_sample = pyro.sample("y_sample", dist.Categorical(logits=y))
        _ = pyro.sample("y_hat", dist.Categorical(logits=mean), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        modules = []
        for (i, spline) in enumerate(self.splines):
            modules.append(pyro.module(f"spline{i}", spline))

    def edge_mask(self, explainer):
        sample = self.flow_dist.rsample(torch.Size([10000, ]))
        sample = sample.sigmoid() if self.sigmoid else sample
        sample = sample.clamp(0, 1)
        return sample.mean(dim=0)


