from torch_geometric.nn import GNNExplainer
from BaseSearcher import BaseSearcher


class GNNExplainerSearcher(BaseSearcher):
    def __init__(self, name: str, epochs):
        self.name = name
        self.epochs = epochs

    def search(self, X, y, explainer, **train_hparams):
        exp = GNNExplainer(explainer.model, epochs=self.epochs)
        _, edge_mask = exp.explain_node(explainer.node_idx, explainer.x,
                                        explainer.edge_index)
        return edge_mask

    def run_name(self):
        return self.name
