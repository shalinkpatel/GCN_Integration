import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("SERGIO")
from SERGIO.sergio import sergio

def SERGIOSimulate(sparsity_level):
    print("=" * 70)
    print("***** Sparsity level = {} *****".format(sparsity_level))
    sim = sergio(number_genes=100, number_bins=2, number_sc=300, noise_params=1, decays=0.8, sampling_state=15, noise_type='dpd')
    sim.build_graph(
        input_file_taregts='SERGIO/data_sets/De-noised_100G_2T_300cPerT_4_DS1/Interaction_cID_4.txt',
        input_file_regs='SERGIO/data_sets/De-noised_100G_2T_300cPerT_4_DS1/Regs_cID_4.txt',
        shared_coop_state=2
    )
    sim.simulate()
    expr = sim.getExpressions()
    
    """
    Add outlier genes
    """
    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
    
    """
    Add Library Size Effect
    """
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.6, scale = 0.4)
    
    """
    # Add Dropouts
    # """
    binary_ind = sim.dropout_indicator(expr_O_L, shape=6.5, percentile=100-sparsity_level)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    
    """
    Convert to UMI count
    """
    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
    
    """
    Make a 2d gene expression matrix
    """
    count_matrix = np.concatenate(count_matrix, axis=1).T
    num_cells, num_genes = count_matrix.shape
    pd.DataFrame(
        data=count_matrix,
        index=["cell{}".format(i) for i in range(num_cells)],
        columns=["gene{}".format(i) for i in range(num_genes)]
    ).to_csv("./generated_data/100gene-2groups-{}sparsity.csv".format(sparsity_level))

    """
    Save cell-type labels
    """
    torch.save(torch.tensor([i // 300 for i in range(num_cells)]), './final_data/100gene-2groups-labels.pt')
   
if __name__ == '__main__':
    SERGIOSimulate(sparsity_level = 1)
    SERGIOSimulate(sparsity_level = 5)
    SERGIOSimulate(sparsity_level = 10)
    SERGIOSimulate(sparsity_level = 15)
    SERGIOSimulate(sparsity_level = 20)
