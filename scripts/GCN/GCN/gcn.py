import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cell_line', nargs=1, type=str, help='cell line to run on')
parser.add_argument('--name', nargs=1, type=str, help='name of dataset')
parser.add_argument('--shuffle', nargs=1, type=str, help='Permute nodes or not')
parser.add_argument('--randomize', nargs=1, type=str, help='Randomize HMs or not')
args = parser.parse_args()

cl = args.cell_line[0]
name = args.name[0]
shuffle = args.shuffle[0] == 'True'
randomize = args.randomize[0] == 'True'

print("%s\t%s\t%s\n" % (cl, name, shuffle))

import dgl
import networkx as nx
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing

with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_class1_unflattened.pickle' % cl, 'rb') as f:
    data = pickle.load(f)
    data_embedding = data.x.reshape(data.x.shape[0], 1, data.x.shape[1], data.x.shape[2]).float()

#with open('/gpfs_home/spate116/data/spate116/GCN/%s/data/data_embedding.pickle' % cl, 'rb') as f:
    #data_embedding = pickle.load(f)

edges = data.edge_index.t()
adj = list(map(lambda x: (x[0].item(), x[1].item()), edges))

graph = nx.read_gpickle("/gpfs_home/spate116/data/spate116/GCN/%s/data/graph.pickle" % cl)
weights = [x[2] for x in graph.edges.data('weight')]
robust_scaler = preprocessing.RobustScaler()
weights = np.ndarray.flatten(robust_scaler.fit_transform(np.array(weights).reshape(-1, 1)))

y = torch.tensor(list(map(lambda x: x[0], data.y)), dtype=torch.long)

G = dgl.DGLGraph(adj)

if shuffle:
    import random
    random.seed(30)
    idx = list(range(len(G.nodes)))
    random.shuffle(idx)
    data_embedding = data_embedding[idx]
    y = y[idx]
    
if randomize:
    m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    data_embedding = m.sample(sample_shape=torch.Size([data.x.shape[0], 1, data.x.shape[1], data.x.shape[2]])).float().reshape(data.x.shape[0], 1, data.x.shape[1], data.x.shape[2])

G.ndata['feat'] = data_embedding
G.ndata['expr'] = y
G.edata['weight'] = torch.tensor(weights, dtype=torch.float)

from dgl.nn.pytorch import SAGEConv, GraphConv, ChebConv, TAGConv, GATConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='pool', activation=torch.tanh)
        self.conv2 = ChebConv(hidden_size, hidden_size1, 4)
        self.conv3 = TAGConv(hidden_size1, hidden_size2, activation=F.leaky_relu, k=3)
        self.conv4 = SAGEConv(hidden_size2, hidden_size3, aggregator_type='pool', activation=torch.tanh)
        self.conv5 = ChebConv(hidden_size3, hidden_size4, 4)
        self.conv6 = TAGConv(hidden_size4, num_classes, activation=F.leaky_relu, k=3)
        x = 100
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
        h = self.conv1(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv3(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv4(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv5(g, h)
        h = torch.tanh(h)
        h = F.dropout(h, training=self.training)
        h = self.conv6(g, h)
        h = torch.sigmoid(h)
        return h

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

device = torch.device('cuda')
def train_model(net, graph, epochs, learning_rate, train_mask, test_mask):
    model = net.to(device)
    graph = graph.to(device)
    samples = len(graph.ndata['expr'])
    correct = graph.ndata['expr'].cpu().numpy().tolist()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    losses_train = []
    losses_test = []
    best_auc = -1
    correct_pred = [G.ndata['expr'].cpu().numpy()[i] for i in test_mask]

    weight_one = sum(G.ndata['expr'].cpu().numpy().tolist())/samples
    weight = torch.tensor([weight_one, 1-weight_one]).to(device)

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        model.train()
        logits = model(graph, graph.ndata['feat'].float())

        loss = F.cross_entropy(logits[train_mask], graph.ndata['expr'][train_mask], weight=weight)
        loss_test = F.cross_entropy(logits[test_mask], graph.ndata['expr'][test_mask], weight=weight)
        losses_train.append(loss.item())
        losses_test.append(loss_test.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        pred = list(map(lambda x: np.argmax(x, axis = 0), torch.exp(F.log_softmax(logits, 1)).cpu().detach().numpy()))
        auc = roc_auc_score(correct_pred, [pred[i] for i in test_mask], average='weighted')
        best_auc = best_auc if best_auc > auc else auc

        pbar.set_description('Best Test AUC: %.4f | Test AUC: %.4f | Test Loss: %.4f' % (best_auc, auc, loss_test.item()))

    return losses_train, losses_test, model, best_auc

import random
random.seed(30)
idx = list(range(len(G.nodes)))
random.shuffle(idx)
train_mask = idx[:10000]
test_mask = idx[10000:]

net = GCN(94, 1250, 750, 250, 100, 25, 2)

# 0.8220
losses_train, losses_test, model, best_auc = train_model(net, G, 500, 0.0001, train_mask, test_mask)

print('Best AUC: %.8f' % best_auc)

model.eval()
logits = model(G, G.ndata["feat"].float())
pred = list(map(lambda x: np.argmax(x, axis = 0), torch.exp(F.log_softmax(logits, 1)).cpu().detach().numpy()))

print("Test Acc: %.8f" % (sum(np.array([G.ndata['expr'].cpu().numpy()[i] for i in test_mask]) == np.array([pred[i] for i in test_mask]))/len(test_mask)))
print("Total Acc: %.8f" % (sum(G.ndata['expr'].cpu().numpy() == pred)/len(G.nodes)))
print()
print("Test AUC: %.8f" % (roc_auc_score([G.ndata['expr'].cpu().numpy()[i] for i in test_mask], [pred[i] for i in test_mask], average='weighted')))
print("Total AUC: %.8f" % (roc_auc_score(G.ndata['expr'].cpu().numpy(), pred, average='weighted')))

torch.save(torch.tensor(logits, dtype=torch.float), "/gpfs_home/spate116/data/spate116/GCN/%s/data/pred.pt" % cl)
torch.save(model.state_dict(), "/gpfs_home/spate116/data/spate116/GCN/%s/res/best_run.model" % cl)
