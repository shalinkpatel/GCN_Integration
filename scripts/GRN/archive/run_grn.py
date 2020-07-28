import pandas as pd
import genie3

ex_matrix = pd.read_csv('~/data/spate116/GCN/E123/K562_expression_matrix_imputed.tsv', sep='\t').transpose()

genie3.get_link_list(genie3.GENIE3(ex_matrix.to_numpy(), gene_names=list(ex_matrix.columns), ntrees=750, nthreads=31), gene_names=list(ex_matrix.columns), file_name='/gpfs/data/rsingh47/spate116/GCN/E123/K562_GRN.tsv')
