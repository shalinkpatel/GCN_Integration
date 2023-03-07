import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist

from .BaseSampler import BaseSampler


class SpikeSlabSampler(BaseSampler):
    def __init__(self, name: str, alpha1: float, beta1: float):
        super().__init__(name)
        self.alpha = alpha1
        self.beta = beta1
        self.alpha1 = alpha1
        self.beta1 = beta1

    def sample_model(self, X, y, explainer):
        alpha = torch.tensor([self.alpha for _ in range(explainer.N)]).to(explainer.device)
        beta = torch.tensor([self.beta for _ in range(explainer.N)]).to(explainer.device)
        t = pyro.sample("t", dist.Beta(alpha, beta).to_event(1))

        f1 = pyro.sample("f1", dist.Beta(self.alpha1, self.beta1).expand([explainer.N]).to_event(1))
        f2 = pyro.sample("f2", dist.Delta(torch.tensor(1.0)).expand([explainer.N]).to_event(1))
        m = pyro.sample("m", dist.Bernoulli((1 - t) * f1 + t * f2).to_event(1))

        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        alpha = torch.tensor([self.alpha for _ in range(explainer.N)]).to(explainer.device)
        beta = torch.tensor([self.beta for _ in range(explainer.N)]).to(explainer.device)
        t_q1 = pyro.param("t_q1", alpha, constraint=constraints.positive)
        t_q2 = pyro.param("t_q2", beta, constraint=constraints.positive)
        t = pyro.sample("t", dist.Beta(t_q1, t_q2).to_event(1))

        alpha1_q = pyro.param("a1_q", torch.tensor([self.alpha1]).to(explainer.device), constraint=constraints.positive)
        beta1_q = pyro.param("b1_q", torch.tensor([self.beta1]).to(explainer.device), constraint=constraints.positive)

        d_q = pyro.param("d_q", torch.tensor(1.0), constraint=constraints.unit_interval)

        f1 = pyro.sample("f1", dist.Beta(alpha1_q, beta1_q).expand([explainer.N]).to_event(1))
        f2 = pyro.sample("f2", dist.Delta(d_q).expand([explainer.N]).to_event(1))

        m = pyro.sample("m", dist.Bernoulli((1 - t) * f1 + t * f2).to_event(1))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()))

    def edge_mask(self, explainer):
        t1 = pyro.param("t_q1")
        t2 = pyro.param("t_q2")
        t = dist.Beta(t1, t2).mean
        
        alpha1 = pyro.param("a1_q")
        beta1 = pyro.param("b1_q")
        
        imp = (1 - t) * dist.Beta(alpha1, beta1).mean + t
        return imp

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_alpha-{self.alpha1}_beta-{self.beta1}"
