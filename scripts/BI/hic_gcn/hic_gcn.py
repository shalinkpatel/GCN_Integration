from scipy.sparse import load_npz
import os
import argparse
from datetime import datetime, date
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
import networkx as nx
from torch_geometric import utils, data
from torch_geometric.nn.models import GNNExplainer
from torch_geometric.utils import k_hop_subgraph, from_networkx
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from src.sage_conv_cat import SAGEConvCat
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from math import sqrt
from matplotlib.colors import Normalize
import matplotlib.font_manager as font_manager

cell_line = 'E116'
base_path = '/gpfs/home/spate116/singhlab/GCN_Integration/scripts/BI/hic_gcn/'
save_dir = os.path.join(base_path, 'src', 'data', cell_line, 'saved_runs')
hic_sparse_mat_file = os.path.join(base_path, 'src', 'data', cell_line, 'hic_sparse.npz')
np_nodes_lab_genes_file = os.path.join(base_path, 'src', 'data', cell_line, 'np_nodes_lab_genes.npy')
np_hmods_norm_all_file = os.path.join(base_path, 'src', 'data', cell_line, 'np_hmods_norm.npy') 
load_model_file = os.path.join(base_path, 'src', 'data', cell_line, 'saved_runs', 'model_2020-12-17-at-19-24-36.pt')

mat  = load_npz(hic_sparse_mat_file)
allNodes_hms = np.load(np_hmods_norm_all_file)
geneNodes_labs = np.load(np_nodes_lab_genes_file)

hms = allNodes_hms[:, 1:] #only includes features, not node ids
allNodes = allNodes_hms[:, 0].astype(int)
geneNodes = geneNodes_labs[:, -2].astype(int)
geneLabs = geneNodes_labs[:, -1].astype(int)

allLabs = 2*np.ones(np.shape(allNodes))
allLabs[geneNodes] = geneLabs
x = torch.tensor(hms).float().reshape(-1, 5)
y = torch.tensor(allLabs).long()

def to_cpu_npy(x):
    if type(x) == list:
        new_x = []
        for element in x:
            new_x.append(element.cpu().numpy())
    else:
        new_x = x.cpu().detach().numpy()
    return new_x

class GCN(nn.Module):
    def __init__(self, num_feat, num_graph_conv_layers, graph_conv_embed_sizes, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines model class
​
        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_embed_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Number of classes to be predicted (2)
​
        Returns
        -------
        None.
​
        '''
        
        super(GCN, self).__init__()
        
        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_lin_layers = num_lin_layers
        self.dropout_value = 0
        
        if self.num_graph_conv_layers == 1:
            self.conv1 = SAGEConvCat(num_feat, graph_conv_embed_sizes)
        elif self.num_graph_conv_layers == 2:
            self.conv1 = SAGEConvCat(num_feat, graph_conv_embed_sizes)
            self.conv2 = SAGEConvCat(graph_conv_embed_sizes, graph_conv_embed_sizes)
        elif self.num_graph_conv_layers == 3:
            self.conv1 = SAGEConvCat(num_feat, graph_conv_embed_sizes)
            self.conv2 = SAGEConvCat(graph_conv_embed_sizes, graph_conv_embed_sizes)
            self.conv3 = SAGEConvCat(graph_conv_embed_sizes, graph_conv_embed_sizes)
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(graph_conv_embed_sizes, num_classes)
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(graph_conv_embed_sizes, lin_hidden_sizes)
            self.lin2 = nn.Linear(lin_hidden_sizes, num_classes)
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(graph_conv_embed_sizes, lin_hidden_sizes)
            self.lin2 = nn.Linear(lin_hidden_sizes, lin_hidden_sizes)
            self.lin3 = nn.Linear(lin_hidden_sizes, num_classes)
    
        self.loss_calc = nn.CrossEntropyLoss()
        self.torch_softmax = nn.Softmax(dim=1)       
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function.
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout
​
        Returns
        -------
        scores [tensor]: Un-normalized class scores
​
        '''
        if self.num_graph_conv_layers == 1:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 2:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 3:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
            h = self.conv3(h, edge_index)
            h = torch.relu(h)
                    
        dropout_value = 0.5
        
        if self.num_lin_layers == 1:
            scores = self.lin1(h)
        elif self.num_lin_layers == 2:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = F.dropout(scores, p = dropout_value, training=train_status)
            scores = self.lin2(scores)
        elif self.num_lin_layers == 3:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = F.dropout(scores, p = dropout_value, training=train_status)
            scores = self.lin2(scores)
            scores = torch.relu(scores)
            scores = self.lin3(scores)
        
        return scores
    
    def loss(self, scores, labels):
        '''
        Calculates cross-entropy loss
        
        Parameters
        ----------
        scores [tensor]: Un-normalized class scores from forward function
        labels [tensor]: Class labels for nodes
​
        Returns
        -------
        xent_loss [tensor]: Cross-entropy loss
​
        '''
        xent_loss = self.loss_calc(scores, labels)
        return xent_loss
    
    def calc_softmax_pred(self, scores):
        '''
        Calculates softmax scores and predicted classes
​
        Parameters
        ----------
        scores [tensor]: Un-normalized class scores
​
        Returns
        -------
        softmax [tensor]: Probability for each class
        predicted [tensor]: Predicted class
​
        '''
        
        softmax = self.torch_softmax(scores)
        
        predicted = torch.argmax(softmax, 1)
        
        return softmax, predicted
cuda_flag = torch.cuda.is_available()
if cuda_flag:  
  dev = "cuda" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

def train_model(log=False):
    extract = utils.from_scipy_sparse_matrix(mat)
    
    G = data.Data(edge_index = extract[0], edge_attr = extract[1], x = x, y = y)
    edge_index = G.edge_index
    
    num_feat = 5
    num_graph_conv_layers = 2    
    graph_conv_embed_sizes = 256
    num_lin_layers = 3
    lin_hidden_sizes = 256
    num_classes = 2
    
    model = GCN(num_feat, num_graph_conv_layers, graph_conv_embed_sizes, num_lin_layers, lin_hidden_sizes, num_classes)
    model.load_state_dict(torch.load(load_model_file, map_location=torch.device('cpu')))
    model.eval()
 
    return model, x, y, edge_index

def extract_subgraph(node_idx, num_hops, edge_index):
    if num_hops == 0:
        sub = [True for i in range(edge_index.shape[1])]
        for i in range(edge_index.shape[1]):
            sub[i] = sub[i] and (edge_index[0, i] == node_idx or edge_index[1, i] == node_idx)
        return edge_index[:, sub], node_idx
    else:
        nodes, new_edge_index, mapping, _ = k_hop_subgraph(node_idx, num_hops, edge_index)
        return new_edge_index, node_idx

def run_model(edge_mask, edge_index, model, node_idx):
    edge_index_1 = edge_index[:, torch.tensor(edge_mask).to(device).bool()]
    out = model(x, edge_index_1).detach().cpu()
    return F.softmax(out[node_idx]).numpy()

if __name__ == '__main__':
    model, x, y, edge_index = train_model()
    explainer = GNNExplainer(model, epochs=1000, num_hops=1)
    node_idx = 60561
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=y)
    plt.savefig('/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/hic_gcn/explain/node%d.png' % node_idx, dpi=300)