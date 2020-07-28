import os
import pandas as pd
import timeit
from arboreto.algo import grnboost2, genie3
from arboreto.utils import load_tf_names
from distributed import LocalCluster, Client

ex_matrix = pd.read_csv('~/scratch/net1_expression_data.tsv', sep='\t')
print(ex_matrix.shape)

tf_names = pd.read_csv('~/scratch/net1_transcription_factors.tsv', sep='\t', header=None).values.flatten().tolist()

local_cluster = LocalCluster(n_workers=15,
                             threads_per_worker=1)
custom_client = Client(local_cluster)

def run_boost():
	return grnboost2(expression_data=ex_matrix.to_numpy(),
                    gene_names=ex_matrix.columns,
                    client_or_address=custom_client)

print(timeit.timeit(run_boost, number=1))
