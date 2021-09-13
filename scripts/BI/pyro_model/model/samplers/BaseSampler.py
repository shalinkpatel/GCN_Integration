from abc import abstractmethod


class BaseSampler():
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sample_model(self, X, y, explainer):
        pass

    @abstractmethod
    def sample_guide(self, X, y, explainer):
        pass

    @abstractmethod
    def edge_mask(self, explainer):
        pass
