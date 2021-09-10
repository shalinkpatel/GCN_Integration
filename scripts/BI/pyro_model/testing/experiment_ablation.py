# coding: utf-8

# # Ablation study
# 
# This notebook can be used to replicate the ablation study presented in the replication study. Despite the ablation study in the replication paper itself only being performed using the tree-cycles dataset and the PGExpainer, the code in this notebook can be used for the other datasets and GNNExplainer as well. 
# 
# The ablation study is first loads the configuration for a single model and a single datasets. By manipulating this configuration file iteratively all possible permutations are checked. 
# 
# **Be aware that running this notebook can take very long to run.** If a quicker version of the script is needed, we recommend replacing the first part of the 3rd code cell with the second part. This should still show the interesting parts of the evaluation. 

# In[1]:


from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication


# In[2]:


_dataset = 'treecycles' # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
_explainer = 'biexplainer' # One of: pgexplainer, gnnexplainer


# PGExplainer
config_path = f"./ExplanationEvaluation/configs/replication/explainers/{_explainer}/{_dataset}.json"

config = Selector(config_path).args.explainer


# In[3]:


# Permutations
#coef_size = [10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.0001]
#coef_entr = [10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.0001]

### QUICKER VERSION
#coef_size = [1.0, 0.01, 0.0001]
#coef_entr = [1.0, 0.01, 0.0001]
#config.seeds = [0]

### BI Version
coef_size = [1.0]
coef_entr = [1.0]
config.seeds = [0]


# In[4]:


results = []
    
for size in coef_size:
    for entropy in coef_entr:
        print(size, entropy)
        interim_resuts = {}

        config.reg_size = size
        config.reg_ent = entropy
        config.temps = 1.0
        config.sample_bias = 0.0

        (auc, std), _ = replication(config, run_qual=False, results_store=False)

        interim_resuts["AUC"] = auc
        interim_resuts["std"] = std
        
        res = {
            'size' : size,
            'entropy' : entropy,
            'auc': auc,
            'std': std
        }
        results.append(res)


# In[5]:


for r in results:
    print(r)

