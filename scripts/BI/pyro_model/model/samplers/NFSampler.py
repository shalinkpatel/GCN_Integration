import pyro
import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T

from .BaseSampler import BaseSampler


class NFSampler(BaseSampler):
    def __init__(self, name, splines: int, sigmoid: bool, lambd: float, p: float, device: torch.device):
        super().__init__(name)
        self.sigmoid = sigmoid
        self.splines_n = splines
        self.device = device

        self.init = False

        self.lambd = lambd
        self.p = p

    def _init_node(self, N):
        self.base_dist = dist.Normal(torch.zeros(N).to(self.device), torch.ones(N).to(self.device))
        self.splines = []
        for _ in range(self.splines_n):
            self.splines.append(T.spline(N).to(self.device))
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.splines)

    def sample_model(self, X, y, explainer):
        if not self.init:
            self._init_node(explainer.edge_index_adj.shape[1])
        m_sub = self.flow_dist.rsample(torch.Size([250, ]))
        if self.sigmoid:
            m_sub = m_sub.sigmoid().clamp(0, 1).mean(dim=0)
        else:
            m_sub = m_sub.clamp(0, 1).mean(dim=0)
        m = pyro.sample("m", dist.Bernoulli(m_sub).to_event(1))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        if not self.init:
            self._init_node(explainer.edge_index_adj.shape[1])
        modules = []
        for (i, spline) in enumerate(self.splines):
            modules.append(pyro.module(f"spline{i}", spline))
        m_sub = self.flow_dist.rsample(torch.Size([250, ]))
        if self.sigmoid:
            m_sub = m_sub.sigmoid().clamp(0, 1).mean(dim=0)
        else:
            m_sub = m_sub.clamp(0, 1).mean(dim=0)
        m = pyro.sample("m", dist.Bernoulli(m_sub).to_event(1))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean), obs=y_sample)

    def edge_mask(self, explainer):
        sample = self.flow_dist.rsample(torch.Size([10000, ]))
        sample = sample.sigmoid() if self.sigmoid else sample
        sample = sample.clamp(0, 1)
        return sample.mean(dim=0)

    def L(self, p):
        sample = self.flow_dist.rsample(torch.Size([250, ]))
        sample = sample.sigmoid() if self.sigmoid else sample.clamp(0, 1)
        sample = sample.pow(p)
        sample = sample / sample.max()
        return sample.mean()

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args) + self.lambd * self.L(self.p)

    def run_name(self):
        return f"{self.name}_splines-{len(self.splines)}_sig-{self.sigmoid}_lambd-{self.lambd}_p-{self.p}"
