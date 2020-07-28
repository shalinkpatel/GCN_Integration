import numpy as np
import pandas as pd
from scipy import stats
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from distributed import Client, LocalCluster
from dask import dataframe
import dask.dataframe as dd
import dask.bag as db

import argparse

def top_targets(num, target):
    return network[network['target'] == target][0:num]

def top_std(z_score, target):
    curr = network[network['target'] == target]
    imp = curr['importance'].to_numpy()
    imp = stats.zscore(imp)
    return curr[imp >= z_score]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line', nargs=1, type=str, help='cell line to run on')
    parser.add_argument('--name', nargs=1, type=str, help='name of dataset')
    args = parser.parse_args()

    cl = args.cell_line[0]
    name = args.name[0]

    cluster = LocalCluster()
    client = Client(cluster)
    network = pd.read_csv('/gpfs_home/spate116/data/spate116/GCN/%s/%s_GRN.tsv' % (cl, name), sep='\t', header=0, names=['TF', 'target', 'importance'])
    gene_set = list(set(network['target'].to_list()))
    split = db.from_sequence(gene_set, npartitions=10)
    connections = split.map(lambda x: [x, top_std(1, x)])
    result_network = connections.compute()
    np.save('/gpfs_home/spate116/data/spate116/GCN/%s/%s_GRN_1_STD' % (cl, name), result_network)
