import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist

from .BaseSampler import BaseSampler


class BetaBernoulliSampler(BaseSampler):
    def __init__(self, name: str, alpha: float, beta: float):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta

    def sample_model(self, X, y, explainer):
        alpha = torch.tensor([self.alpha for _ in range(explainer.N)]).to(explainer.device)
        beta = torch.tensor([self.beta for _ in range(explainer.N)]).to(explainer.device)
        f = pyro.sample("f", dist.Beta(alpha, beta).to_event(1))
        m = pyro.sample("m", dist.Bernoulli(f).to_event(1))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        alpha = torch.tensor([self.alpha for _ in range(explainer.N)]).to(explainer.device)
        beta = torch.tensor([self.beta for _ in range(explainer.N)]).to(explainer.device)
        alpha_q = pyro.param("alpha_q", alpha, constraint=constraints.positive)
        beta_q = pyro.param("beta_q", beta, constraint=constraints.positive)
        f = pyro.sample("f", dist.Beta(alpha_q, beta_q).to_event(1))
        m = pyro.sample("m", dist.Bernoulli(f).to_event(1))
        mean = explainer.model(X, explainer.edge_index_adj[:, m == 1])[explainer.mapping].reshape(-1).exp()
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()))

    def edge_mask(self, explainer):
        alpha = pyro.param('alpha_q')
        beta = pyro.param('beta_q')
        return dist.Beta(alpha, beta).mean

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_alpha-{self.alpha}_beta-{self.beta}"
