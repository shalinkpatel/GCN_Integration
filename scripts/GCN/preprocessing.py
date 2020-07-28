import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cell_line', nargs=1, type=str, help='cell line to run on')
parser.add_argument('--name', nargs=1, type=str, help='name of dataset')
args = parser.parse_args()

cl = args.cell_line[0]
name = args.name[0]

network = np.load('/gpfs_home/spate116/data/spate116/GCN/%s/%s_GRN_1_STD.npy' % (cl, name), allow_pickle=True)

import networkx as nx

G = nx.DiGraph()
for target in network:
    for index, row in target[1].iterrows():
        G.add_edge(row['TF'], row['target'], weight=row['importance'])

RPKM = pd.read_csv('/gpfs_home/spate116/data/spate116/GCN/57epigenomes.RPKM.pc', sep='\t', header=0)

columns = RPKM.columns[1:len(RPKM.columns)]
RPKM = RPKM.drop('E128', axis=1)
RPKM = RPKM.set_axis(columns, axis=1)

x = RPKM.values #returns a numpy array
robust_scaler = preprocessing.RobustScaler()
x_scaled = robust_scaler.fit_transform(x)
RPKM = pd.DataFrame(x_scaled, columns=RPKM.columns, index=RPKM.index)
exp = RPKM[cl]

conversion = pd.read_csv('/gpfs_home/spate116/data/spate116/GCN/%s/conv_info.tsv' % cl, sep='\t')

converted = {}
for elt in exp.index:
    idx = conversion.index[conversion['V1'] == elt].tolist()
    val = conversion.loc[idx]['V7']
    if not val.empty:
        converted[elt] = val.tolist()[0]

epigenetic = pd.read_csv('/gpfs_home/spate116/data/spate116/GCN/data/%s/classification/train.csv' % cl, index_col=None, names=['GeneID', 'Bin ID', 'H3K27me3 count', 'H3K36me3 count', 'H3K4me1 count', 'H3K4me3 count', 'H3K9me3 counts', 'Classifications'], dtype={'GeneID' : str}).append(
    pd.read_csv('/gpfs_home/spate116/data/spate116/GCN/data/%s/classification/valid.csv' % cl, index_col=None, names=['GeneID', 'Bin ID', 'H3K27me3 count', 'H3K36me3 count', 'H3K4me1 count', 'H3K4me3 count', 'H3K9me3 counts', 'Classifications'], dtype={'GeneID' : str}), ignore_index=True).append(
    pd.read_csv('/gpfs_home/spate116/data/spate116/GCN/data/%s/classification/test.csv' % cl, index_col=None, names=['GeneID', 'Bin ID', 'H3K27me3 count', 'H3K36me3 count', 'H3K4me1 count', 'H3K4me3 count', 'H3K9me3 counts', 'Classifications'], dtype={'GeneID' : str}), ignore_index=True)[['GeneID', 'H3K27me3 count', 'H3K36me3 count', 'H3K4me1 count', 'H3K4me3 count', 'H3K9me3 counts', 'Classifications']]

epigenetic_list = list(set(list(map(lambda x: 'ENSG00000' + x, epigenetic['GeneID']))))
inverted_converted = {v: k for k, v in converted.items()}

converted_exp = {}
for key in converted:
    converted_exp[converted[key]] = exp[key]

unsupported = []
for name in list(G.nodes):
    if name not in list(converted_exp):
        unsupported.append(name)

for node in unsupported:
    G.remove_node(node)

final_nodes = {}
for node in list(G.nodes):
    final_nodes[node] = converted_exp[node]

original_id = [inverted_converted[k] for k in final_nodes]

epigenetic['GeneID'] = list(map(lambda x: 'ENSG00000' + x, epigenetic['GeneID']))

idx = list(map(lambda x: x in original_id, epigenetic['GeneID'].to_list()))
epigenetic = epigenetic[idx]

further_unsupported = []
for elt in original_id:
    if epigenetic[epigenetic['GeneID'] == elt].shape[0] == 200:
        further_unsupported.append(elt)

remove_from_graph_2 = [converted[x] for x in further_unsupported]

for elt in further_unsupported:
    original_id.remove(elt)

for elt in remove_from_graph_2:
    G.remove_node(elt)

epigentic_batched = {}
class_y = {}
for elt in original_id:
    epigentic_batched[converted[elt]] = epigenetic[epigenetic['GeneID'] == elt][['H3K27me3 count',
                                                        'H3K36me3 count',
                                                        'H3K4me1 count',
                                                        'H3K4me3 count',
                                                        'H3K9me3 counts']].to_numpy()
    class_y[converted[elt]] = epigenetic[epigenetic['GeneID'] == elt][['Classifications']].to_numpy()[0]

node_id_to_name = {}
node_name_to_id = {}
for i in range(len(original_id)):
    node_id_to_name[i] = converted[original_id[i]]
    node_name_to_id[converted[original_id[i]]] = i

G = nx.relabel_nodes(G, node_name_to_id)

import torch
edges = torch.tensor(list(nx.edges(G)), dtype=torch.long).t()

X = []
X_unflattened = []
y = []
y_class = []
for i in range(len(original_id)):
    X.append(np.ndarray.flatten(epigentic_batched[node_id_to_name[i]]))
    X_unflattened.append(epigentic_batched[node_id_to_name[i]])
    y.append([converted_exp[node_id_to_name[i]]])
    y_class.append(class_y[node_id_to_name[i]])

X = np.array(X)
X_unflattened = np.array(X_unflattened)
y = np.array(y)
y_class = np.array(y_class)

X = torch.tensor(X, dtype=torch.double)
X_unflattened = torch.tensor(X_unflattened, dtype=torch.double)
y = torch.tensor(y, dtype=torch.double)
y_class = torch.tensor(y_class, dtype=torch.int)

nx.write_gpickle(G, "/gpfs_home/spate116/data/spate116/GCN/%s/data/graph.pickle" % cl)

from torch_geometric.data import Data
data = Data(x=X, y=y, edge_index=edges)
data_class = Data(x=X, y=y_class, edge_index=edges)
data_class_unflattened = Data(x=X_unflattened, y=y_class, edge_index=edges)

import pickle
with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data1.pickle' % cl, 'wb') as f:
    pickle.dump(data, f)

with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_class1.pickle' % cl, 'wb') as f:
    pickle.dump(data_class, f)

with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_class1_unflattened.pickle' % cl, 'wb') as f:
    pickle.dump(data_class_unflattened, f)
    
with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/conv.pickle' % cl, 'wb') as f:
    pickle.dump(node_id_to_name, f)
