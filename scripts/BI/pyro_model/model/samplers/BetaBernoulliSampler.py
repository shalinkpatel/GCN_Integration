import pyro
from BaseSampler import BaseSampler

class BetaBernoulliSampler(BaseSampler):
    def __init__(self, name: str, alpha: float, beta: float):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta
        