import networkx as nx
from synthetic import gen_syn4
import pickle
import numpy as np
from utils import featgen

G, labels, _ = gen_syn4(feature_generator=featgen.ConstFeatureGen(np.ones(10, dtype=float)))
nx.write_gpickle(G, 'data/syn4_G.pickle')

with open('data/syn4_lab.pickle', 'wb') as f:
    pickle.dump(labels, f)