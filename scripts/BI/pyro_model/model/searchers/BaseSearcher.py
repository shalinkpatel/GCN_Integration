from abc import abstractmethod


class BaseSearcher:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search(self, X, y, explainer, **train_hparams):
        pass

    @abstractmethod
    def run_name(self):
        pass
