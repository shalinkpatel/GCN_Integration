from abc import abstractmethod

class BaseSampler():
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sample_model(self, X, y):
        pass

    @abstractmethod
    def sample_guide(self, X, y):
        pass

    @abstractmethod
    def edge_mask(self):
        pass