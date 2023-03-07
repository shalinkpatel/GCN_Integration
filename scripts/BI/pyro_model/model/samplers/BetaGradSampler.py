import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist

from torch_geometric.nn.models.explainer import clear_masks, set_masks

from .BaseSampler import BaseSampler


class BetaGradSampler(BaseSampler):
    def __init__(self, name: str, alpha: float, beta: float):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta

    def sample_model(self, X, y, explainer):
        alpha = torch.tensor([self.alpha for _ in range(explainer.N)]).to(explainer.device)
        beta = torch.tensor([self.beta for _ in range(explainer.N)]).to(explainer.device)
        f = pyro.sample("f", dist.Beta(alpha, beta).to_event(1))
        set_masks(explainer.model, torch.nn.Parameter(f), explainer.edge_index_adj, False)
        mean = explainer.model(X, explainer.edge_index_adj)[explainer.mapping].reshape(-1).exp()
        clear_masks(explainer.model)
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        return pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        alpha = torch.tensor([self.alpha for _ in range(explainer.N)]).to(explainer.device)
        beta = torch.tensor([self.beta for _ in range(explainer.N)]).to(explainer.device)
        alpha_q = pyro.param("alpha_q", alpha, constraint=constraints.positive)
        beta_q = pyro.param("beta_q", beta, constraint=constraints.positive)
        f = pyro.sample("f", dist.Beta(alpha_q, beta_q).to_event(1))
        set_masks(explainer.model, torch.nn.Parameter(f), explainer.edge_index_adj, False)
        mean = explainer.model(X, explainer.edge_index_adj)[explainer.mapping].reshape(-1).exp()
        clear_masks(explainer.model)
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()))

    def edge_mask(self, explainer):
        return explainer.samples['f'].mean(dim=0)

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_alpha-{self.alpha}_beta-{self.beta}"
