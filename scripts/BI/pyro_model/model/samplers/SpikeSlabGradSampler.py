import pyro
import torch
from pyro.distributions import constraints
import pyro.distributions as dist

from torch_geometric.nn.models.explainer import clear_masks, set_masks

from .BaseSampler import BaseSampler


class SpikeSlabGradSampler(BaseSampler):
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
        f = (1 - t) * f1 + t

        set_masks(explainer.model, torch.nn.Parameter(f), explainer.edge_index_adj, False)
        mean = explainer.model(X, explainer.edge_index_adj)[explainer.mapping].reshape(-1).exp()
        clear_masks(explainer.model)
        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        return pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()), obs=y_sample)

    def sample_guide(self, X, y, explainer):
        alpha = torch.tensor([self.alpha for _ in range(explainer.N)]).to(explainer.device)
        beta = torch.tensor([self.beta for _ in range(explainer.N)]).to(explainer.device)
        t_q1 = pyro.param("t_q1", alpha, constraint=constraints.positive)
        t_q2 = pyro.param("t_q2", beta, constraint=constraints.positive)
        t = pyro.sample("t", dist.Beta(t_q1, t_q2).to_event(1))

        alpha1_q = pyro.param("a1_q", torch.tensor([self.alpha1]).to(explainer.device), constraint=constraints.positive)
        beta1_q = pyro.param("b1_q", torch.tensor([self.beta1]).to(explainer.device), constraint=constraints.positive)

        f1 = pyro.sample("f1", dist.Beta(alpha1_q, beta1_q).expand([explainer.N]).to_event(1))
        f = (1 - t) * f1 + t

        set_masks(explainer.model, torch.nn.Parameter(f), explainer.edge_index_adj, False)
        mean = explainer.model(X, explainer.edge_index_adj)[explainer.mapping].reshape(-1).exp()
        clear_masks(explainer.model)

        y_sample = pyro.sample("y_sample", dist.Categorical(probs=y/y.sum()))
        _ = pyro.sample("y_hat", dist.Categorical(probs=mean/mean.sum()))

    def edge_mask(self, explainer):
        t = explainer.samples['t']
        return (explainer.samples['f1'] * (1-t) + t).mean(dim=0)

    def loss_fn(self, model, guide, *args, **kwargs):
        return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, *args)

    def run_name(self):
        return f"{self.name}_alpha-{self.alpha1}_beta-{self.beta1}"
