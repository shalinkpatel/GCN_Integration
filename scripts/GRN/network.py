import os
import pandas as pd
import argparse
from dask.distributed import Client
from distributed import Client, LocalCluster

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line', nargs=1, type=str, help='cell line to run on')
    parser.add_argument('--name', nargs=1, type=str, help='name of dataset')
    args = parser.parse_args()

    cl = args.cell_line[0]
    name = args.name[0]

    from arboreto.algo import grnboost2, genie3
    from arboreto.utils import load_tf_names

    ex_matrix = pd.read_csv('~/data/spate116/GCN/%s/%s_expression_matrix_imputed.tsv' % (cl, name), sep='\t').transpose()

    cluster = LocalCluster()
    client = Client(cluster)
    print('here')
    network = grnboost2(expression_data=ex_matrix.to_numpy(), gene_names=ex_matrix.columns, client_or_address=client)
    network.to_csv('~/data/spate116/GCN/%s/%s_GRN.tsv' % (cl, name), sep='\t', header=True, index=False)
    client.close()
    cluster.close()
