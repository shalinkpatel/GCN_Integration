from random import choices
import torch
from typing import List


class TreeWalk:
    def __init__(self):
        self.probs = {}
        for p in [0.4, 0.675]:
            other_out = (1 - p) / 5
            self.probs[p] = [
                [0, p / 2, (1 - p) / 4, (1 - p) / 4, p / 2, (1 - p) / 4, (1 - p) / 4],
                [p, 0, other_out, other_out, other_out, other_out, other_out],
                [other_out, p, 0, other_out, other_out, other_out, other_out],
                [other_out, p, other_out, 0, other_out, other_out, other_out],
                [p, other_out, other_out, other_out, 0, other_out, other_out],
                [other_out, other_out, other_out, other_out, p, 0, other_out],
                [other_out, other_out, other_out, other_out, other_out, p, 0]
            ]

    def sample_counts(self, p: float) -> List[int]:
        nodes = list(range(7))
        counts = [0 for _ in nodes]
        curr = choices(nodes, k=1)[0]
        for _ in range(75):
            prev = curr
            curr = choices(nodes, self.probs[p][curr], k=1)[0]
            if self.probs[p][prev][curr] >= 0.2:
                counts[curr] += 1
        return counts

    def generate_dataset(self):
        gt_grn = torch.tensor([[0, 0, 2, 3, 5, 6, 4, 1], [1, 4, 1, 1, 4, 5, 0, 0]]).long()
        comp_graph = [[], []]
        for i in range(7):
            for j in range(7):
                if i != j:
                    comp_graph[0].append(i)
                    comp_graph[1].append(j)
        comp_graph = torch.tensor(comp_graph).long()

        x = [self.sample_counts(0.4) for _ in range(500)]
        y = [0 for _ in range(500)]

        x += [self.sample_counts(0.675) for _ in range(500)]
        y += [1 for _ in range(500)]

        x = torch.tensor(x).T.float()
        y = torch.tensor(y).long()

        torch.save(gt_grn, "gt_grn.pt")
        torch.save(comp_graph, "comp_graph.pt")
        torch.save(x, 'x.pt')
        torch.save(y, "y.pt")


if __name__ == '__main__':
    walker = TreeWalk()
    walker.generate_dataset()
