import sys
sys.path.append("..")

from models import GCN_classification
import torch
import pickle
import numpy as np

model = GCN_classification(6, 2, [6, 256, 256], 3, [256, 256, 256, 2], 2)
model.load_state_dict(torch.load('data/E116/model_2021-06-26-at-05-46-03.pt', map_location=torch.device('cpu')))
model.eval()

with open('data/E116/node_imp_score_stats_top5_1700genes_ct_2021-10-13-at-15-06-45.pkl', 'rb') as f:
    data = pickle.load(f)