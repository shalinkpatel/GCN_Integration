from abc import abstractmethod

from torch_geometric.nn import MessagePassing
import torch


class BaseSampler:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sample_model(self, X, y, explainer):
        pass

    @abstractmethod
    def sample_guide(self, X, y, explainer):
        pass

    @abstractmethod
    def edge_mask(self, explainer) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_fn(self, model, guide, *args, **kwargs):
        pass

    @abstractmethod
    def run_name(self):
        pass
