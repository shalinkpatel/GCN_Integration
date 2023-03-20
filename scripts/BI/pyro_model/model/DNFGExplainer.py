import pyro
import torch
import torch.nn.functional as F
import pyro.distributions as dist
import pyro.distributions.transforms as T
from tqdm import tqdm


class DNFGExplainer:
    def __init__(self, model: torch.nn.Module, splines: int, X: torch.Tensor, G: torch.Tensor, device: torch.device):
        self.model = model
        self.n_splines = splines
        self.X = X
        self.G = G
        self.target = self.model(self.X, self.G)

        self.ne = G.shape[1]

        self.base_dist = dist.Normal(torch.zeros(self.ne).to(device), torch.ones(self.ne).to(device))
        self.splines = []
        self.params_l = []
        for _ in range(self.n_splines):
            self.splines.append(T.spline(self.ne).to(device))
            self.params_l += self.splines[-1].parameters()
        self.params = torch.nn.ParameterList(self.params_l)
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.splines)

    def forward(self, model, X, G, m):
        preds = model(X, G, edge_weight=m)
        return preds

    def edge_mask(self):
        return self.flow_dist.rsample(torch.Size([100, ])).sigmoid().mean(dim=0)

    def train(self, epochs: int, lr: float, log: bool):
        optimizer = torch.optim.Adam(self.params, lr=lr)
        pbar = range(epochs)
        if log:
            pbar = tqdm(pbar)

        best_loss = 1e20
        for epoch in pbar:
            optimizer.zero_grad()
            m = self.edge_mask()
            preds = self.forward(self.model, self.X.detach(), self.G.detach(), m)
            kl = F.kl_div(preds, self.target, log_target=True)
            reg = m.mean()
            loss = kl + 0.1*reg
            loss_val = loss.detach().cpu().item()
            loss.backward()
            optimizer.step()
            self.flow_dist.clear_cache()

            if loss_val < best_loss:
                best_loss = loss_val
            if log:
                pbar.set_description(f"Epoch {epoch} Best Loss {best_loss}")

    def clean(self):
        cpu = torch.device('cpu')
        for spl in self.splines:
            spl.to(cpu)
        for p in self.params_l:
            p.to(cpu)
        self.params.to(cpu)

        del self.base_dist
        del self.splines
        del self.params_l
        del self.params
        del self.flow_dist
