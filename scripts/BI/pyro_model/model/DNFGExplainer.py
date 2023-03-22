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
        with torch.no_grad():
            self.target = self.model(self.X, self.G)

        self.ne = G.shape[1]

        self.base_dist = dist.Beta(0.95 * torch.ones(self.ne).to(device), 0.95 * torch.ones(self.ne).to(device))
        self.splines = []
        self.params_l = []
        for _ in range(self.n_splines):
            self.splines.append(T.spline(self.ne).to(device))
            self.params_l += self.splines[-1].parameters()
        self.params = torch.nn.ParameterList(self.params_l)
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.splines)

    def forward(self):
        m = self.flow_dist.rsample().sigmoid()
        preds = self.model(self.X, self.G, edge_weight=m)
        return preds, m

    def edge_mask(self):
        return self.flow_dist.sample(torch.Size([250, ])).sigmoid().mean(dim=0)

    def train(self, epochs: int, lr: float):
        optimizer = torch.optim.Adam(self.params, lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds, m = self.forward()
            kl = F.kl_div(preds, self.target, log_target=True)
            reg = m.mean()
            loss = kl + 0.01*reg
            loss.backward()
            optimizer.step()
            self.flow_dist.clear_cache()

    def clean(self):
        cpu = torch.device('cpu')
        for spl in self.splines:
            spl = spl.to(cpu)
        for p in self.params_l:
            p = p.to(cpu)
        self.params = self.params.to(cpu)
        self.X = self.X.to(cpu)
        self.G = self.G.to(cpu)

        del self.base_dist
        del self.splines
        del self.params_l
        del self.params
        del self.flow_dist
        del self.X
        del self.G
