import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist

from .BaseSampler import BaseSampler


class SpikeSlabSampler(BaseSampler):
    def __init__(self, name: str, theta: float, alpha1: float, beta1: float, alpha2: float, beta2: float):
        super().__init__(name)
        self.theta = theta
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    def sample_model(self, X, y, explainer):
        theta = torch.tensor([self.theta for _ in range(explainer.N)]).to(explainer.device)
        t = pyro.sample("t", dist.Bernoulli(theta).to_event(1))

        f1 = pyro.sample("f1", dist.Beta(self.alpha1, self.beta1).expand([explainer.N]).to_event(1))
        f2 = pyro.sample("f2", dist.Beta(self.alpha2, self.beta2).expand([explainer.N]).to_event(1))

        m = pyro.sample("m", dist.Bernoulli((1 - t) * f1 + t * f2).to_event(1))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean))

    def sample_guide(self, X, y, explainer):
        theta = torch.tensor([self.theta for _ in range(explainer.N)]).to(explainer.device)
        t_q = pyro.param("t_q", theta, constraint=constraints.unit_interval)
        t = pyro.sample("t", dist.Bernoulli(t_q).to_event(1))

        alpha1_q = pyro.param("a1_q", torch.tensor([self.alpha1]).to(explainer.device), constraint=constraints.positive)
        beta1_q = pyro.param("b1_q", torch.tensor([self.beta1]).to(explainer.device), constraint=constraints.positive)
        f1 = pyro.sample("f1", dist.Beta(alpha1_q, beta1_q).expand([explainer.N]).to_event(1))

        alpha2_q = pyro.param("a2_q", torch.tensor([self.alpha2]).to(explainer.device), constraint=constraints.positive)
        beta2_q = pyro.param("b2_q", torch.tensor([self.beta2]).to(explainer.device), constraint=constraints.positive)
        f2 = pyro.sample("f2", dist.Beta(alpha2_q, beta2_q).expand([explainer.N]).to_event(1))

        m = pyro.sample("m", dist.Bernoulli((1 - t) * f1 + t * f2).to_event(1))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()))

    def edge_mask(self, explainer):
        t = pyro.param("t_q")
        
        alpha1 = pyro.param("a1_q")
        beta1 = pyro.param("b1_q")
        
        alpha2 = pyro.param("a2_q")
        beta2 = pyro.param("b2_q")
        
        imp = (1 - t) * dist.Beta(alpha1, beta1).mean + t * dist.Beta(alpha2, beta2).mean
        return imp

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_theta_{self.theta}_alpha-{self.alpha1}-{self.alpha2}_beta-{self.beta1}-{self.beta2}"
