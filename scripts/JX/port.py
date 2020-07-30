from ordered_set import OrderedSet
from six.moves import cPickle as pickle 
from collections import defaultdict
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix

import numpy as np
import torch
import torch_geometric
import networkx as nx

from torch_geometric.nn import SAGEConv, ChebConv, TAGConv, GATConv, ARMAConv
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

import random

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, hidden_size1, num_classes, k):
        super(GCN, self).__init__()
        self.conv1 = TAGConv(in_feats, hidden_size, K = k)
        self.conv2 = TAGConv(hidden_size, hidden_size1, K = k)
        self.conv3 = TAGConv(hidden_size1, num_classes, K = k)
        x = 10
        self.encoder = nn.Sequential(
            nn.Conv2d(1, x, (3, 3)),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(x, 2*x, (3, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(2*x, 1, (3, 2)),
        )

    def forward(self, g, inputs):
        h = self.encoder(inputs).reshape(-1, 94)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv1(h, g.edge_index)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(h, g.edge_index)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv3(h, g.edge_index)
        h = F.softmax(h, dim=1)
        return h
    
def train_model(net, graph, epochs, learning_rate, train_mask, test_mask, mask):
    device = torch.device('cuda')
    model = net.to(device)
    graph = graph.to(device)
    samples = len(graph.y)
    correct = graph.y.cpu().numpy().tolist()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    losses_train = []
    losses_test = []
    best_auc = -1
    correct_pred = [graph.y.cpu().numpy()[i] for i in test_mask]

    weight_one = sum(graph.y.cpu().numpy().tolist())/samples
    weight = torch.tensor([weight_one, 1-weight_one]).to(device)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        model.train()
        logits = model(graph, graph.x.float())[mask]

        loss = F.cross_entropy(logits[train_mask], graph.y[train_mask], weight=weight)
        loss_test = F.cross_entropy(logits[test_mask], graph.y[test_mask], weight=weight)
        losses_train.append(loss.item())
        losses_test.append(loss_test.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        pred = list(map(lambda x: np.argmax(x, axis = 0), torch.exp(F.log_softmax(logits, 1)).cpu().detach().numpy()))
        auc = roc_auc_score(correct_pred, [pred[i] for i in test_mask], average='weighted')
        best_auc = best_auc if best_auc > auc else auc

        pbar.set_description('Best Test AUC: %.4f | Train Loss: %.4f | Test Loss: %.4f' % (best_auc, loss.item(), loss_test.item()))

    return best_auc

def run_sim(cl, epochs, k):
    mat = load_npz('/gpfs/data/rsingh47/jbigness/data/%s/hic_sparse_vcsqrt_oe_edge_v7.npz' % cl)
    hms = np.load('/gpfs/data/rsingh47/jbigness/data/%s/np_hmods_norm_vcsqrt_oe_edge_v7.npy' % cl)
    labs = np.load('/gpfs/data/rsingh47/jbigness/data/%s/np_nodes_lab_genes_vcsqrt_oe_edge_v7.npy' % cl)
    
    mask = torch.tensor(labs[:,-1]).long()
    X = torch.tensor(hms[:mat.shape[0]]).float().reshape(-1, 1, 100, 5)
    y = torch.tensor(labs[:,-2]).long()
    
    extract = torch_geometric.utils.from_scipy_sparse_matrix(mat)
    data = torch_geometric.data.Data(edge_index = extract[0], edge_attr = extract[1], x = X, y = y)
    G = data
    
    random.seed(30)
    idx = list(range(G.y.shape[0]))
    random.shuffle(idx)
    train_mask = idx[:10000]
    test_mask = idx[10000:]
    
    net = GCN(94, 75, 50, 2, k = k)
    return train_model(net, G, epochs, 0.01, train_mask, test_mask, mask) 