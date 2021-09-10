import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import pyro
import pyro.distributions as dist
from torch_geometric.utils import k_hop_subgraph, to_networkx
import torch.distributions.constraints as constraints
from torch_geometric.data import Data
import networkx as nx
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine
from copy import copy
from math import sqrt
import pyro.distributions.transforms as T

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge

class BayesianExplainer(BaseExplainer):
    def __init__(self, model_to_explain, graphs, features, task, k: int = 3,  epochs: int = 10000, lr: float = 2, lambd: float = 1.5e-11, window: int = 500, p = 1.25, sharp: float = 1e-12, splines: int = 12, sigmoid = False):
        super().__init__(model_to_explain, graphs, features, task)
        self.model = model_to_explain
        self.edge_index = graphs
        self.x = features
        self.epochs = epochs
        self.lr = lr
        self.lambd = lambd
        self.window = window
        self.p = p
        self.sharp = sharp
        self.splines_n = splines
        self.sigmoid = sigmoid
        self.k = k
        self.log = False

    def setup_node(self, node_idx: int):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        self.node_idx = node_idx
        self.subset, self.edge_index_adj, self.mapping, self.edge_mask_hard = k_hop_subgraph(
            self.node_idx, self.k, self.edge_index, relabel_nodes=True)
        self.x_adj = self.x[self.subset]
        self.device = device

        with torch.no_grad():
            self.preds = self.model(self.x, self.edge_index_adj)
        
        self.N = self.edge_index_adj.size(1)
        self.base_dist = dist.Normal(torch.zeros(self.N).to(device), torch.ones(self.N).to(device))
        self.splines = []
        for _ in range(self.splines_n):
            self.splines.append(T.spline(self.N).to(device))
        self.flow_dist = dist.TransformedDistribution(self.base_dist,self.splines)
    
    def sample_model(self, X, y):
        if self.sigmoid:
            m = pyro.sample("m", dist.Bernoulli(self.flow_dist.rsample(torch.Size([250,])).sigmoid().clamp(0, 1).mean(dim=0)).to_event(1))
        else:
            m = pyro.sample("m", dist.Bernoulli(self.flow_dist.rsample(torch.Size([250,])).clamp(0, 1).mean(dim=0)).to_event(1))
        mean = self.model(X, self.edge_index_adj[:, m == 1])[self.mapping].reshape(-1)
        #sigma = (self.sharp * torch.eye(mean.size(0))).to(self.device)
        #y_sample = pyro.sample("y", dist.MultivariateNormal(mean, sigma), obs = y)
        y_sample = pyro.sample("y_sample", dist.Categorical(logits=y))
        y_hat = pyro.sample("y_hat", dist.Categorical(logits=mean), obs=y_sample)
    
    def sample_guide(self, X, y):
        modules = []
        for (i, spline) in enumerate(self.splines):
            modules.append(pyro.module("spline%i" % i, spline))
        
    def ma(self, l, window):
        cumsum, moving_aves = [0], []

        for i, x in enumerate(l, 1):
            cumsum.append(cumsum[i-1] + x)
            if i>=window:
                moving_ave = (cumsum[i] - cumsum[i-window])/window
                #can do stuff with moving_ave here
                moving_aves.append(moving_ave)
        return moving_aves
    
    def L(self, p):
        sample = self.flow_dist.rsample(torch.Size([250,]))
        if self.sigmoid:
            return sample.sigmoid().pow(p).mean()
        else:
            return sample.clamp(0, 1).pow(p).mean()

    def train(self):
        pyro.get_param_store().clear()
        params = []
        for spline in self.splines:
            params += spline.parameters()
        optimizer = torch.optim.Adam(params, self.lr)
        loss_fn = lambda model, guide, X, y: pyro.infer.Trace_ELBO().differentiable_loss(model, guide, X, y) * self.sharp + self.lambd * self.L(self.p)

        n_steps = self.epochs
        # do gradient steps
        if self.log:
            pbar = tqdm(range(n_steps))
        else:
            pbar = range(n_steps)
        elbos = []
        for step in pbar:
            loss = loss_fn(self.sample_model, self.sample_guide, self.x_adj, self.preds[self.mapping[0]])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            elbos.append(loss.cpu().detach().item())
            avgs = self.ma(elbos, self.window)
            if step >= self.window:
                disp = avgs[-1]
            else:
                disp = loss.cpu().detach().item()
            if self.log:
                pbar.set_description("Loss -> %.4f" % disp)
        return avgs
    
    def edge_mask(self):
        sample = self.flow_dist.rsample(torch.Size([10000,]))
        if self.sigmoid:
            return sample.sigmoid().mean(dim=0)
        else:
            return sample.clamp(0, 1).mean(dim=0)

    def prepare(self, args):
        return

    def explain(self, index):
        self.setup_node(index)
        self.train()
        return self.edge_index_adj, self.edge_mask()
