import os
import pandas as pd

from arboreto.algo import grnboost2, genie3
from arboreto.utils import load_tf_names

ex_matrix = pd.read_csv('~/scratch/HEPG2_expression_matrix_imputed.tsv', sep='\t').transpose()

from dask_jobqueue import SLURMCluster
from distributed import Client

cluster = SLURMCluster(
    queue='bigmem',
    cores=31,
    processes=31,
    walltime='48:00:00',
    memory="400 GB"
)

print(cluster.job_script())

print(cluster.status)

cluster.scale(jobs=1)

from dask.distributed import Client
client = Client(cluster)

import time
time.sleep(30)

print(client.dashboard_link)

network = grnboost2(expression_data=ex_matrix.to_numpy(),
                    gene_names=ex_matrix.columns,
                    client_or_address=client)

network.to_csv('~/data/spate116/GCN/HEPG2_GRN.tsv', sep='\t', header=True, index=False)

client.close()
cluster.close()
