from ordered_set import OrderedSet
from six.moves import cPickle as pickle 
from collections import defaultdict
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix

import numpy as np
import torch
import torch_geometric
import networkx as nx

from torch_geometric.nn import SAGEConv, ChebConv, TAGConv, GATConv, ARMAConv, BatchNorm
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

import random

from torch_geometric.data import ClusterData, ClusterLoader

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, hidden_size1, hidden_size2, hidden_size3, num_classes, conv):
        super(GCN, self).__init__()
        self.conv1 = conv(in_feats, hidden_size)
        self.bn1 = BatchNorm(hidden_size)
        self.conv2 = conv(hidden_size, hidden_size1)
        self.bn2 = BatchNorm(hidden_size1)
        self.conv3 = conv(hidden_size1, hidden_size2)
        self.bn3 = BatchNorm(hidden_size2)
        self.conv4 = conv(hidden_size2, hidden_size3)
        self.bn4 = BatchNorm(hidden_size3)
        self.conv5 = conv(hidden_size3, num_classes)
        self.bn5 = BatchNorm(num_classes)
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
        h = self.bn1(h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(h, g.edge_index)
        h = self.bn2(h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv3(h, g.edge_index)
        h = self.bn3(h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv4(h, g.edge_index)
        h = self.bn4(h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv5(h, g.edge_index)
        h = self.bn5(h)
        h = F.softmax(h, dim=1)
        return h
    
def train_model(net, data_loader, epochs, learning_rate, train_mask, test_mask, mask):
    device = torch.device('cuda')
    model = net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    losses_train = []
    losses_test = []
    auc_l = []    
    best_auc = -1

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        logits = []
        y = []
        for d in data_loader:
            d = d.to(device)
            model.train()
            logits.append(model(d, d.x.float()))
            y.append(d.y)
        
        logits = torch.cat(logits, dim=0).to(device)
        y = torch.cat(y, dim=0)
        mask = (y != -1)
        
        logits = logits[mask]
        y = y[mask]
        
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss_test = F.cross_entropy(logits[test_mask], y[test_mask])
        losses_train.append(loss.item())
        losses_test.append(loss_test.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        pred = list(map(lambda x: np.argmax(x, axis = 0), torch.exp(F.log_softmax(logits, 1)).cpu().detach().numpy()))
        auc = roc_auc_score(y[test_mask].cpu().numpy(), [pred[i] for i in test_mask], average='weighted')
        auc_l.append(auc)
        best_auc = best_auc if best_auc > auc else auc

        pbar.set_description('Best Test AUC: %.4f | Train Loss: %.4f | Test Loss: %.4f' % (best_auc, loss.item(), loss_test.item()))

    return best_auc, losses_test, losses_train, auc_l

def run_sim(cl, batches, layer):
    layer_dict = {
        'arma': ARMAConv,
        'sage': SAGEConv,
        'tag': TAGConv
    }
    mat = load_npz('/gpfs/data/rsingh47/jbigness/data/%s/hic_sparse_vcsqrt_oe_edge_v9.npz' % cl)
    hms = np.load('/gpfs/data/rsingh47/jbigness/data/%s/np_hmods_norm_vcsqrt_oe_edge_v9.npy' % cl)
    labs = np.load('/gpfs/data/rsingh47/jbigness/data/%s/np_nodes_lab_genes_vcsqrt_oe_edge_v9.npy' % cl)
    
    print('Data Loaded')
    
    mask = torch.tensor(labs[:,-1]).long()
    loc = {}
    for i in range(labs[:, -1].shape[0]):
        loc[labs[i, -1]] = i
    y = []
    for i in range(mat.shape[0]):
        y.append(labs[loc[i],-2]) if i in mask else y.append(-1)
    y = torch.tensor(y).long()
    extract = torch_geometric.utils.from_scipy_sparse_matrix(mat)
    G = torch_geometric.data.Data(edge_index = extract[0], 
                                  edge_attr = extract[1], 
                                  x = torch.tensor(hms).float().reshape(-1, 1, 100, 5), 
                                  y = y)
    
    cluster_data = ClusterData(G, num_parts=batches, recursive=False)
    train_loader = ClusterLoader(cluster_data, batch_size=2, shuffle=False,
                             num_workers=0)
    
    print('Data Clustered')
    
    random.seed(30)
    idx = list(range(labs.shape[0] - 1))
    random.shuffle(idx)
    train_mask = idx[:10000]
    test_mask = idx[10000:]
    
    net = GCN(94, 500, 300, 250, 50, 2, layer_dict[layer])
    return train_model(net, train_loader, 1500, 0.0005, train_mask, test_mask, mask)

