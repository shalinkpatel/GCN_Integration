import torch
import pyro.distributions as dist
import pyro
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
import torch.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


class BetaExplainer:
    def __init__(self, model: torch.nn.Module, X: torch.Tensor, G: torch.Tensor, device: torch.device):
        self.model = model
        self.X = X
        self.G = G
        with torch.no_grad():
            self.target = self.model(self.X, self.G)

        self.ne = G.shape[1]
        self.N = X.shape[0]
        self.obs = 100
        self.device = device

    def model_p(self, ys):
        alpha = 2.0 * torch.ones(self.N).to(self.device)
        beta = 6.0 * torch.ones(self.N).to(self.device)
        alpha_edges = alpha[self.G[0, :]]
        beta_edges = beta[self.G[1, :]]
        m = pyro.sample("mask", dist.Beta(alpha_edges, beta_edges).to_event(1))
        set_masks(self.model, m, self.G, False)
        preds = self.model(self.X, self.G).exp()
        with pyro.plate("data_loop"):
            pyro.sample("obs", dist.Categorical(preds), obs=ys)

    def guide(self, ys):
        alpha = pyro.param("alpha_q", 2.0 * torch.ones(self.N).to(self.device), constraint=constraints.positive)
        beta = pyro.param("beta_q", 6.0 * torch.ones(self.N).to(self.device), constraint=constraints.positive)
        alpha_edges = alpha[self.G[0, :]]
        beta_edges = beta[self.G[1, :]]
        pyro.sample("mask", dist.Beta(alpha_edges, beta_edges).to_event(1))

    def train(self, epochs: int, lr: float = 0.0005):
        adam_params = {"lr": lr, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)
        svi = SVI(self.model_p, self.guide, optimizer, loss=Trace_ELBO())

        for _ in range(epochs):
            ys = torch.distributions.categorical.Categorical(self.target.exp()).sample(torch.Size([self.obs]))
            svi.step(ys)

        clear_masks(self.model)

    def edge_mask(self):
        m = torch.distributions.beta.Beta(pyro.param("alpha_q").detach(), pyro.param("beta_q").detach()).sample(torch.Size([250]))
        return m.mean(dim=0)
