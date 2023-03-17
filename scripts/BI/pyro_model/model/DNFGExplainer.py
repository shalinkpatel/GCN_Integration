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
        self.params = []
        for _ in range(self.n_splines):
            self.splines.append(T.spline(self.ne).to(device))
            self.params += self.splines[-1].parameters()
        self.params = torch.nn.ParameterList(self.params)
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.splines)

    def forward(self):
        m = self.edge_mask()
        preds = self.model(self.X, self.G, edge_weight=m)
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
            preds = self.forward()
            m = self.edge_mask()
            loss = F.kl_div(preds, self.target, log_target=True) + 0.5 * m.mean()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.flow_dist.clear_cache()

            if loss.item() < best_loss:
                best_loss = loss.item()
            if log:
                pbar.set_description(f"Epoch {epoch} Best Loss {loss}")
