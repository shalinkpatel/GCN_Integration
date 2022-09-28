import torch
import networkx as nx
import pickle
import numpy as np

G = nx.read_gpickle('data/syn3_G.pickle')
with open('data/syn3_lab.pickle', 'rb') as f:
    labels = pickle.load(f)

x = torch.tensor([x[1]['feat'] for x in G.nodes(data=True)])
edge_index = torch.tensor([x for x in G.edges])
edge_index_flipped = edge_index[:, [1, 0]]
edge_index = torch.cat((edge_index, edge_index_flipped))
y = torch.tensor(labels, dtype=torch.long)

print("Data Successfully Loaded")
print(f"x: {x}")
print(f"edge_index: {edge_index}")
print(f"y: {y}")

with open("data/x.npy", "wb") as f:
    np.save(f, x.numpy())

with open("data/g.npy", "wb") as f:
    np.save(f, edge_index.numpy())

with open("data/y.npy", "wb") as f:
    np.save(f, y.numpy())